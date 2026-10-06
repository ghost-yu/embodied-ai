import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from beartype import beartype
from typing import Tuple

from einops import rearrange, repeat

# helpers


def exists(val):
    if isinstance(val, str):
        return val != ""
    return val is not None


def default(val, d):
    return val if exists(val) else d


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def l2norm(t):
    return F.normalize(t, dim=-1)


# bias-less layernorm, being used in more recent T5s, PaLM, also in @borisdayma 's experiments shared with me
# greater stability


class LayerNorm(nn.Module):
    """
    A custom LayerNorm implementation without bias.

    Args:
        dim (int): The dimension of the input.
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        """
        Apply layer normalization to the input.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# feedforward


class GEGLU(nn.Module):
    """
    Gated Error Linear Units (GELU) activation function.
    """

    def forward(self, x):
        """
        Apply GELU activation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output after GELU activation.
        """
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, dropout=0.0):
    """
    Create a feedforward neural network.

    Args:
        dim (int): The input dimension.
        mult (int): The multiplier for the inner dimension.
        dropout (float): The dropout rate.

    Returns:
        nn.Sequential: The feedforward neural network.
    """
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


# PEG - position generating module


class PEG(nn.Module):
    """
    Position Encoding Generator (PEG) module.

    Args:
        dim (int): The input dimension.
        causal (bool): Whether to use causal convolution.
    """

    def __init__(self, dim, causal=False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    @beartype
    def forward(
        self, x, shape_pattern: str = "", shape: Tuple[int, int, int, int] = None
    ):
        """
        Apply position encoding.

        Args:
            x (torch.Tensor): The input tensor.
            shape_pattern (str): The shape pattern for rearrangement.
            shape (Tuple[int, int, int, int]): The shape of the input.

        Returns:
            torch.Tensor: The tensor with position encoding applied.
        """
        needs_shape = x.ndim == 3
        assert not (
            needs_shape and not (exists(shape_pattern) and exists(shape))
        ), "If x is 3D, shape_pattern and shape must be provided"

        if needs_shape:
            b, t, h, w = shape
            x = rearrange(x, f"{shape_pattern} -> b t h w d", b=b, t=t, h=h, w=w)
        x = rearrange(x, "b ... d -> b d ...")

        frame_padding = (2, 0) if self.causal else (1, 1)

        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value=0.0)
        x = self.dsconv(x)
        x = rearrange(x, "b d ... -> b ... d")

        if needs_shape:
            x = rearrange(x, "b ... d -> b (...) d")

        return rearrange(
            x,
            f"b (t h w) d -> {shape_pattern}",
            b=shape[0],
            t=shape[1],
            h=shape[2],
            w=shape[3],
        )


# attention


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        dim (int): The input dimension.
        dim_context (int): The context dimension.
        dim_head (int): The dimension of each attention head.
        heads (int): The number of attention heads.
        causal (bool): Whether to use causal attention.
        num_null_kv (int): The number of null key-value pairs.
        norm_context (bool): Whether to apply normalization to the context.
        dropout (float): The dropout rate.
        scale (float): The scale factor for attention scores.
    """

    def __init__(
        self,
        dim,
        dim_context=None,
        dim_head=64,
        heads=8,
        causal=False,
        num_null_kv=0,
        norm_context=True,
        dropout=0.0,
        scale=8,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads=heads)

        self.attn_dropout = nn.Dropout(dropout)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.num_null_kv = num_null_kv
        if self.num_null_kv > 0:
            self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))
        else:
            self.null_kv = None

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None, context=None, attn_bias=None):
        """
        Apply multi-head attention.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The attention mask.
            context (torch.Tensor): The context tensor.
            attn_bias (torch.Tensor): The attention bias.

        Returns:
            torch.Tensor: The output after applying attention.
        """
        batch, device, dtype = x.shape[0], x.device, x.dtype

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)
        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        if self.num_null_kv > 0:
            nk, nv = repeat(
                self.null_kv, "h (n r) d -> b h n r d", b=batch, r=2
            ).unbind(dim=-2)

            k = torch.cat((nk, k), dim=-2)
            v = torch.cat((nv, v), dim=-2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        i, j = sim.shape[-2:]

        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.0)
            sim = sim + attn_bias

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            sim = sim + self.rel_pos_bias(sim)

            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(
                j - i + 1
            )

            sim = sim.masked_fill(causal_mask, -torch.inf)

        attn = torch.nan_to_num(sim.softmax(dim=-1), nan=0.0)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# alibi positional bias for extrapolation


class AlibiPositionalBias(nn.Module):
    """
    Alibi Positional Bias for attention mechanisms.

    Args:
        heads (int): The number of attention heads.
    """

    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, "h -> h 1 1")
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    def get_bias(self, i, j, device):
        """
        Generate the positional bias.

        Args:
            i (int): The query sequence length.
            j (int): The key sequence length.
            device (torch.device): The device to use.

        Returns:
            torch.Tensor: The positional bias.
        """
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(
            rearrange(j_arange, "j -> 1 1 j") - rearrange(i_arange, "i -> 1 i 1")
        )
        return bias

    @staticmethod
    def _get_slopes(heads):
        """
        Calculate the slopes for the Alibi positional bias.

        Args:
            heads (int): The number of attention heads.

        Returns:
            List[float]: The slopes for each head.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : heads - closest_power_of_2
            ]
        )

    def forward(self, sim):
        """
        Apply the Alibi positional bias to the attention scores.

        Args:
            sim (torch.Tensor): The attention scores.

        Returns:
            torch.Tensor: The positional bias.
        """
        h, i, j, device = *sim.shape[-3:], sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer("bias", bias, persistent=False)

        return self.bias


class ContinuousPositionBias(nn.Module):
    """
    Continuous Position Bias for attention mechanisms.

    Args:
        dim (int): The dimension of the position encoding.
        heads (int): The number of attention heads.
        num_dims (int): The number of dimensions for the position encoding.
        layers (int): The number of layers in the position encoding network.
        log_dist (bool): Whether to use logarithmic distance.
        cache_rel_pos (bool): Whether to cache the relative positions.
    """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims=2,  # 2 for images, 3 for video
        layers=2,
        log_dist=True,
        cache_rel_pos=False,
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer("rel_pos", None, persistent=False)

    def forward(self, *dimensions, device=torch.device("cpu")):
        """
        Generate the continuous position bias.

        Args:
            *dimensions: The dimensions of the input.
            device (torch.device): The device to use.

        Returns:
            torch.Tensor: The continuous position bias.
        """

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device=device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing="ij"))
            grid = rearrange(grid, "c ... -> (...) c")
            rel_pos = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer("rel_pos", rel_pos, persistent=False)

        rel_pos = self.rel_pos.float()

        for layer in self.net:
            rel_pos = layer(rel_pos)

        return rearrange(rel_pos, "i j h -> h i j")


# transformer


class Transformer(nn.Module):
    """
    A Transformer module that can perform self-attention and cross-attention.

    Args:
        dim (int): The dimension of the input and output.
        depth (int): The number of transformer layers.
        dim_context (int, optional): The dimension of the context for cross-attention.
        causal (bool): Whether to use causal attention.
        dim_head (int): The dimension of each attention head.
        heads (int): The number of attention heads.
        ff_mult (int): The multiplier for the feed-forward layer dimension.
        attn_scale (float): The scale factor for attention scores.
        peg (bool): Whether to use Position Encoding Generator.
        peg_causal (bool): Whether to use causal PEG.
        attn_num_null_kv (int): The number of null key-value pairs for attention.
        has_cross_attn (bool): Whether to include cross-attention layers.
        attn_dropout (float): Dropout rate for attention.
        ff_dropout (float): Dropout rate for feed-forward layers.
    """

    def __init__(
        self,
        dim,
        *,
        depth,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_scale=8,
        peg=False,
        peg_causal=False,
        attn_num_null_kv=2,
        has_cross_attn=False,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PEG(dim=dim, causal=peg_causal) if peg else None,
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            scale=attn_scale,
                            causal=causal,
                            dropout=attn_dropout,
                        ),
                        (
                            Attention(
                                dim=dim,
                                dim_head=dim_head,
                                dim_context=dim_context,
                                heads=heads,
                                scale=attn_scale,
                                causal=False,
                                num_null_kv=attn_num_null_kv,
                                dropout=attn_dropout,
                            )
                            if has_cross_attn
                            else None
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm_out = LayerNorm(dim)

    @beartype
    def forward(
        self,
        x,
        video_shape_pattern: str = "",
        video_shape: Tuple[int, int, int, int] = None,
        attn_bias=None,
        context=None,
        self_attn_mask=None,
        cross_attn_context_mask=None,
    ):
        """
        Forward pass of the Transformer.

        Args:
            x (torch.Tensor): Input tensor.
            video_shape_pattern (str): Pattern describing the video shape.
            video_shape (Tuple[int, int, int, int]): Shape of the video tensor.
            attn_bias (torch.Tensor, optional): Attention bias tensor.
            context (torch.Tensor, optional): Context tensor for cross-attention.
            self_attn_mask (torch.Tensor, optional): Mask for self-attention.
            cross_attn_context_mask (torch.Tensor, optional): Mask for cross-attention context.

        Returns:
            torch.Tensor: Output tensor after passing through the Transformer.
        """

        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x = peg(x, shape_pattern=video_shape_pattern, shape=video_shape) + x

            x = self_attn(x, attn_bias=attn_bias, mask=self_attn_mask) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context, mask=cross_attn_context_mask) + x

            x = ff(x) + x

        return self.norm_out(x)


class STBlock(nn.Module):
    """
    Spatio-Temporal Block that applies spatial and temporal attention.

    Args:
        dim (int): The dimension of the input and output.
        order (str): The order of spatial and temporal attention ('st' or 'ts').
        causal (bool): Whether to use causal attention for temporal attention.
        peg (bool): Whether to use Position Encoding Generator.
        peg_causal (bool): Whether to use causal PEG.
        dim_head (int): The dimension of each attention head.
        heads (int): The number of attention heads.
        attn_scale (float): The scale factor for attention scores.
        attn_dropout (float): Dropout rate for attention.
        ff_mult (int): The multiplier for the feed-forward layer dimension.
        ff_dropout (float): Dropout rate for feed-forward layers.
    """

    def __init__(
        self,
        dim,
        order="st",
        *,
        causal=False,
        peg=False,
        peg_causal=False,
        dim_head=64,
        heads=8,
        attn_scale=8,
        attn_dropout=0.0,
        ff_mult=4,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.spatial_peg = PEG(dim=dim, causal=peg_causal) if peg else None
        self.spatial_attention = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            scale=attn_scale,
            causal=False,
            dropout=attn_dropout,
        )
        self.spatial_ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.temporal_peg = PEG(dim=dim, causal=peg_causal) if peg else None
        self.temporal_attention = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            scale=attn_scale,
            causal=causal,
            dropout=attn_dropout,
        )
        self.temporal_ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.order = order
        assert self.order in ["st", "ts"]

    def spatial_temporal_forward(
        self,
        x,
        pattern,
        shape,
        spatial_pattern,
        temporal_pattern,
        attn_bias,
        attn_mask,
        mask_pattern,
        spatial_mask_pattern=None,
        termporal_mask_pattern=None,
    ):
        """
        Forward pass for spatial-temporal order.

        Args:
            x (torch.Tensor): Input tensor.
            pattern (str): Pattern describing the input shape.
            shape (Tuple[int, int, int, int]): Shape of the input tensor.
            spatial_pattern (str): Pattern for spatial attention.
            temporal_pattern (str): Pattern for temporal attention.
            attn_bias (torch.Tensor): Attention bias tensor.
            attn_mask (torch.Tensor): Attention mask tensor.
            mask_pattern (str): Pattern describing the mask shape.
            spatial_mask_pattern (str, optional): Pattern for spatial attention mask.
            termporal_mask_pattern (str, optional): Pattern for temporal attention mask.

        Returns:
            torch.Tensor: Output tensor after spatial-temporal attention.
        """
        b, t, h, w = shape

        # Spatial attention
        x = rearrange(x, f"{pattern} -> {spatial_pattern}", b=b, t=t, h=h, w=w)
        if exists(attn_mask):
            attn_mask = rearrange(
                attn_mask,
                f"{mask_pattern} -> {spatial_mask_pattern}",
                b=b,
                t=t,
                h=h,
                w=w,
            )
        x = self.spatial_peg(x, shape_pattern=spatial_pattern, shape=shape) + x
        x = self.spatial_attention(x, attn_bias=attn_bias, mask=attn_mask) + x
        x = self.spatial_ff(x) + x

        # Temporal attention
        x = rearrange(x, f"{spatial_pattern} -> {temporal_pattern}", b=b, t=t, h=h, w=w)
        if exists(attn_mask):
            attn_mask = rearrange(
                attn_mask,
                f"{spatial_mask_pattern} -> {termporal_mask_pattern}",
                b=b,
                t=t,
                h=h,
                w=w,
            )
        x = self.temporal_peg(x, shape_pattern=temporal_pattern, shape=shape) + x
        x = self.temporal_attention(x, mask=attn_mask) + x
        x = self.temporal_ff(x) + x

        x = rearrange(x, f"{temporal_pattern} -> {pattern}", b=b, t=t, h=h, w=w)
        return x

    def temporal_spatial_forward(
        self,
        x,
        pattern,
        shape,
        spatial_pattern,
        temporal_pattern,
        attn_bias,
        attn_mask,
        mask_pattern,
        spatial_mask_pattern=None,
        termporal_mask_pattern=None,
    ):
        """
        Forward pass for temporal-spatial order.

        Args:
            x (torch.Tensor): Input tensor.
            pattern (str): Pattern describing the input shape.
            shape (Tuple[int, int, int, int]): Shape of the input tensor.
            spatial_pattern (str): Pattern for spatial attention.
            temporal_pattern (str): Pattern for temporal attention.
            attn_bias (torch.Tensor): Attention bias tensor.
            attn_mask (torch.Tensor): Attention mask tensor.
            mask_pattern (str): Pattern describing the mask shape.
            spatial_mask_pattern (str, optional): Pattern for spatial attention mask.
            termporal_mask_pattern (str, optional): Pattern for temporal attention mask.

        Returns:
            torch.Tensor: Output tensor after temporal-spatial attention.
        """
        b, t, h, w = shape

        # Temporal attention
        x = rearrange(x, f"{pattern} -> {temporal_pattern}", b=b, t=t, h=h, w=w)
        if exists(attn_mask):
            attn_mask = rearrange(
                attn_mask,
                f"{mask_pattern} -> {termporal_mask_pattern}",
                b=b,
                t=t,
                h=h,
                w=w,
            )
        x = self.temporal_peg(x, shape_pattern=temporal_pattern, shape=shape) + x
        x = self.temporal_attention(x, mask=attn_mask) + x
        x = self.temporal_ff(x) + x

        # Spatial attention
        x = rearrange(x, f"{temporal_pattern} -> {spatial_pattern}", b=b, t=t, h=h, w=w)
        if exists(attn_mask):
            attn_mask = rearrange(
                attn_mask,
                f"{termporal_mask_pattern} -> {spatial_mask_pattern}",
                b=b,
                t=t,
                h=h,
                w=w,
            )
        x = self.spatial_peg(x, shape_pattern=spatial_pattern, shape=shape) + x
        x = self.spatial_attention(x, attn_bias=attn_bias, mask=attn_mask) + x
        x = self.spatial_ff(x) + x

        x = rearrange(x, f"{spatial_pattern} -> {pattern}", b=b, t=t, h=h, w=w)
        return x

    def forward(
        self,
        x,
        pattern,
        shape,
        spatial_pattern,
        temporal_pattern,
        attn_bias=None,
        attn_mask=None,
    ):
        """
        Forward pass of the STBlock.

        Args:
            x (torch.Tensor): Input tensor.
            pattern (str): Pattern describing the input shape.
            shape (Tuple[int, int, int, int]): Shape of the input tensor.
            spatial_pattern (str): Pattern for spatial attention.
            temporal_pattern (str): Pattern for temporal attention.
            attn_bias (torch.Tensor, optional): Attention bias tensor.
            attn_mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after spatio-temporal attention.
        """
        mask_pattern = pattern.replace(" d", "")
        spatial_mask_pattern = spatial_pattern.replace(" d", "")
        temporal_mask_pattern = temporal_pattern.replace(" d", "")
        if self.order == "st":
            return self.spatial_temporal_forward(
                x,
                pattern,
                shape,
                spatial_pattern,
                temporal_pattern,
                attn_bias,
                attn_mask,
                mask_pattern,
                spatial_mask_pattern,
                temporal_mask_pattern,
            )
        elif self.order == "ts":
            return self.temporal_spatial_forward(
                x,
                pattern,
                shape,
                spatial_pattern,
                temporal_pattern,
                attn_bias,
                attn_mask,
                mask_pattern,
                spatial_mask_pattern,
                temporal_mask_pattern,
            )


class STTransformer(nn.Module):
    """
    Spatio-Temporal Transformer that applies multiple STBlocks.

    Args:
        dim (int): The dimension of the input and output.
        num_blocks (int): The number of STBlocks to use.
        order (str): The order of spatial and temporal attention ('st' or 'ts').
        causal (bool): Whether to use causal attention for temporal attention.
        dim_head (int): The dimension of each attention head.
        heads (int): The number of attention heads.
        ff_mult (int): The multiplier for the feed-forward layer dimension.
        attn_scale (float): The scale factor for attention scores.
        peg (bool): Whether to use Position Encoding Generator.
        peg_causal (bool): Whether to use causal PEG.
        attn_dropout (float): Dropout rate for attention.
        ff_dropout (float): Dropout rate for feed-forward layers.
    """

    def __init__(
        self,
        dim,
        *,
        num_blocks,
        order="st",
        causal=False,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_scale=8,
        peg=False,
        peg_causal=False,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_blocks):
            self.layers.append(
                nn.ModuleList(
                    [
                        STBlock(
                            dim=dim,
                            order=order,
                            peg=peg,
                            peg_causal=peg_causal,
                            causal=causal,
                            dim_head=dim_head,
                            heads=heads,
                            attn_scale=attn_scale,
                            attn_dropout=attn_dropout,
                            ff_mult=ff_mult,
                            ff_dropout=ff_dropout,
                        ),
                    ]
                )
            )

        self.norm_out = LayerNorm(dim)

    @beartype
    def forward(
        self,
        x,
        pattern,
        spatial_pattern,
        temporal_pattern,
        video_shape: Tuple[int, int, int, int] = None,
        attn_bias=None,
        self_attn_mask=None,
    ):

        for (sttn_block,) in self.layers:
            x = sttn_block(
                x,
                pattern,
                video_shape,
                spatial_pattern,
                temporal_pattern,
                attn_bias=attn_bias,
                attn_mask=self_attn_mask,
            )
            +x

        return self.norm_out(x)
