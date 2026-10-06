from pathlib import Path

import torch
from torch import nn

from einops.layers.torch import Rearrange
from einops import rearrange

from .vector_quantize import VectorQuantize
from .attention import STTransformer, ContinuousPositionBias

import numpy as np


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def divisible_by(numer, denom):
    return (numer % denom) == 0


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


def cast_tuple(val, l=1):
    return val if isinstance(val, tuple) else (val,) * l


class STViViT(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        codebook_size=1024,
        image_size=64,
        patch_size=4,
        temporal_patch_size=1,
        num_blocks=8,
        wandb_mode="disabled",
        codebook_dim=32,
        dim_head=64,
        heads=8,
        channels=3,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4.0,
        vq_loss_w=1.0,
        recon_loss_w=1.0,
        enable_decoder=True,
    ):
        super(STViViT, self).__init__()

        self.wandb_mode = wandb_mode

        self.vq_loss_w = vq_loss_w
        self.recon_loss_w = recon_loss_w
        self.enable_decoder = enable_decoder

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (
            image_width % patch_width
        ) == 0, "Image dimensions must be divisible by the patch size."

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        # Embedding layers for the first frame and the rest of the frames
        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange(
                "b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_emb = nn.Sequential(
            Rearrange(
                "b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)",
                p1=patch_height,
                p2=patch_width,
                pt=temporal_patch_size,
            ),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim),
        )

        # Transformer configuration
        transformer_kwargs = dict(
            num_blocks=num_blocks,
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=True,
            peg=True,
            peg_causal=True,
            ff_mult=ff_mult,
        )

        # Encoder and decoder transformers
        self.encoder = STTransformer(order="st", **transformer_kwargs)
        if self.enable_decoder:
            self.decoder = STTransformer(order="ts", **transformer_kwargs)

        # Vector Quantization layer
        self.vq = VectorQuantize(
            dim=dim,
            codebook_size=codebook_size,
            learnable_codebook=True,
            ema_update=False,
            use_cosine_sim=True,
            commitment_weight=0.25,
            codebook_dim=codebook_dim,
        )

        # Layers to convert tokens back to pixels
        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange(
                "b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)",
                p1=patch_height,
                p2=patch_width,
            ),
        )
        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height * temporal_patch_size),
            Rearrange(
                "b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)",
                p1=patch_height,
                p2=patch_width,
                pt=temporal_patch_size,
            ),
        )

        # wandb config
        config = {}
        arguments = locals()
        for key in arguments.keys():
            if key not in ["self", "config", "__class__"]:
                config[key] = arguments[key]
        self.config = config

    def trainable_parameters(self):
        """
        Get all trainable parameters of the model.

        Returns:
            list: List of trainable parameters.
        """
        return list(self.parameters())

    def load(self, path):
        """
        Load model weights from a file.

        Args:
            path (str or Path): Path to the saved model weights.
        """
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        self.load_state_dict(pt)

    @property
    def patch_height_width(self):
        """
        Calculates and returns the height and width of image patches.

        Returns:
            tuple: The height and width of the patches.
        """

        return (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )

    def encode(self, tokens):
        """
        Encode input tokens using the encoder.

        Args:
            tokens (torch.Tensor): Input tokens.

        Returns:
            torch.Tensor: Encoded tokens.
        """
        h, w = self.patch_height_width  # patch h,w

        video_shape = tuple(tokens.shape[:-1])
        pattern = "b t h w d"
        spatial_pattern = "(b t) (h w) d"
        temporal_pattern = "(b h w) t d"

        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)

        tokens = self.encoder(
            tokens,
            pattern=pattern,
            spatial_pattern=spatial_pattern,
            temporal_pattern=temporal_pattern,
            video_shape=video_shape,
            attn_bias=attn_bias,
        )

        return tokens

    def decode(self, tokens):
        """
        Decode tokens into a video.

        Args:
            tokens (torch.Tensor): Input tokens.

        Returns:
            torch.Tensor: Reconstructed video.
        """
        assert self.enable_decoder, "Decoder is disabled."
        tokens.shape[0]
        h, w = self.patch_height_width

        # Reshape tokens if necessary
        if tokens.ndim == 3:
            tokens = rearrange(tokens, "b (t h w) d -> b t h w d", h=h, w=w)

        video_shape = tuple(tokens.shape[:-1])

        pattern = "b t h w d"
        spatial_pattern = "(b t) (h w) d"
        temporal_pattern = "(b h w) t d"

        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)

        # Apply decoder
        tokens = self.decoder(
            tokens,
            pattern=pattern,
            spatial_pattern=spatial_pattern,
            temporal_pattern=temporal_pattern,
            video_shape=video_shape,
            attn_bias=attn_bias,
        )

        # Convert tokens to pixels
        first_frame_token, rest_frames_tokens = tokens[:, :1], tokens[:, 1:]

        first_frame = self.to_pixels_first_frame(first_frame_token)
        rest_frames = self.to_pixels(rest_frames_tokens)

        recon_video = torch.cat((first_frame, rest_frames), dim=2)

        return recon_video

    def decode_from_codebook_indices(self, indices):
        """
        Decode video from codebook indices.

        Args:
            indices (torch.Tensor): Tensor of codebook indices.

        Returns:
            torch.Tensor: Decoded video.
        """
        codes = self.vq.codebook[indices]
        projected_out_codes = self.vq.project_out(codes)
        return self.decode(projected_out_codes)

    def forward(self, *args, **kwargs):
        """
        Forward pass of the STViViT model.
        """
        ...
