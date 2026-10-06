import functools
import math

import torch
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from torch import nn

from .components.attention import ContinuousPositionBias, STTransformer
from tools.logger import getLogger

log = getLogger(__name__)

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else (val,) * length


def reduce_mult(arr):
    return functools.reduce(lambda x, y: x * y, arr)


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


# decorators


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# classifier free guidance functions
def uniform(shape, device):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


# tensor helper functions
def logarithm(t, eps=1e-10):
    return torch.log(t + eps)


def apply_bernoulli_mask(mask, prob):
    """
    Apply a Bernoulli mask to an existing mask.

    Args:
        mask (torch.Tensor): The existing mask.
        prob (float): The probability for the Bernoulli distribution.

    Returns:
        torch.Tensor: The mask after applying the Bernoulli mask.
    """
    batch, seq_len, device = *mask.shape, mask.device
    bernoulli_mask = torch.bernoulli(
        torch.full((batch, seq_len), prob, device=device)
    ).bool()

    mask = mask & bernoulli_mask

    return mask


# sampling helpers


def gumbel_noise(t):
    """
    Generate Gumbel noise for a given tensor shape.

    Args:
        t (torch.Tensor): A tensor whose shape will be used for generating noise.

    Returns:
        torch.Tensor: Gumbel noise with the same shape as the input tensor.
    """
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -logarithm(-logarithm(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    """
    Perform Gumbel-softmax sampling on a tensor.

    Args:
        t (torch.Tensor): The input tensor.
        temperature (float): The temperature parameter for sampling. Default is 1.0.
        dim (int): The dimension along which to sample. Default is -1.

    Returns:
        torch.Tensor: The sampled tensor.
    """
    return (t + max(temperature, 1e-10) * gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


def mask_ratio_schedule(ratio, total_unknown, method="cosine"):
    """
    Compute the mask ratio based on a given schedule.

    Args:
        ratio (float): The current ratio.
        total_unknown (int): The total number of unknown tokens.
        method (str): The scheduling method. Default is "cosine".

    Returns:
        torch.Tensor: The computed mask ratio.

    Raises:
        ValueError: If an invalid mask ratio method is provided.
    """
    mask_ratio = None
    if method == "uniform":
        mask_ratio = 1.0 - ratio
    elif "pow" in method:
        exponent = float(method.replace("pow", ""))
        mask_ratio = 1.0 - (ratio**exponent)
    elif method == "cosine":
        mask_ratio = torch.cos(ratio * math.pi / 2)
    elif method == "log":
        mask_ratio = -torch.log2(ratio) / torch.log2(
            torch.tensor(total_unknown, dtype=torch.float32)
        )
    elif method == "exp":
        mask_ratio = 1 - torch.exp2(
            -torch.log2(torch.tensor(total_unknown, dtype=torch.float32)) * (1 - ratio)
        )

    if mask_ratio is None:
        raise ValueError(f"Invalid mask ratio method: {method}")

    mask_ratio = torch.clamp(mask_ratio, 1e-6, 1.0)
    return mask_ratio


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, alpha=1):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DistanceWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, codebook, _lamda=1):
        super(DistanceWeightedCrossEntropyLoss, self).__init__()
        self.pairwise_distances = self._compute_pairwise_distances(codebook)
        self._lamda = _lamda

    def _compute_pairwise_distances(self, x):
        """Computes the pairwise cosine distance matrix for a set of vectors (codebook)."""

        with torch.no_grad():
            # Normalize each vector to unit length (L2 normalization)
            x_normalized = F.normalize(
                x, p=2, dim=1
            )  # Shape: (num_classes, vector_dim)

            # Compute the cosine similarity matrix
            cosine_similarity = torch.mm(
                x_normalized, x_normalized.T
            )  # Shape: (num_classes, num_classes)

            # Convert cosine similarity to cosine distance
            cosine_distance = 1 - cosine_similarity

        return cosine_distance.detach()

    def forward(self, predictions, target_indices):
        """
        Computes the custom cross-entropy loss with inverse distance-based coefficients.

        Args:
            predictions (torch.Tensor): Logits from the model (batch_size, num_classes).
            target_indices (torch.Tensor): Ground truth indices (batch_size,).

        Returns:
            torch.Tensor: The computed loss.
        """

        # Device assignment for compatibility
        device = predictions.device

        # Ensure pairwise_distances is on the correct device
        pairwise_distances = self.pairwise_distances.to(device)

        # Select the distances corresponding to the target indices without looping
        distances = pairwise_distances[target_indices]

        # apply softmax to predictions
        softmax_predictions = F.softmax(predictions, dim=1)

        # Compute the distance-weighted penalty
        distance_weighted_penalty = torch.mean(
            torch.sum(distances * softmax_predictions, dim=1)
        )

        # debug("distance_weighted_penalty", distance_weighted_penalty)

        # Compute the final loss
        loss = self._lamda * distance_weighted_penalty

        return loss


class MaskGIT(nn.Module):
    """
    MaskGIT model for video token prediction with token distance loss for training.

    This model uses a transformer architecture to predict video tokens based on
    masked input and action tokens.

    Attributes:
        dim (int): The dimension of the model.
        action_dim (int): The dimension of the action tokens.
        image_size (tuple): The size of the input images.
        patch_size (tuple): The size of the patches.
        mask_id (int): The ID used for masked tokens.
        unconditional (bool): Whether the model is unconditional.
        token_emb (nn.Embedding): Embedding layer for tokens.
        pos_emb (nn.Embedding): Embedding layer for positions.
        gradient_shrink_alpha (float): Alpha value for gradient shrinking.
        continuous_pos_bias (ContinuousPositionBias): Continuous position bias module.
        transformer (STTransformer): The main transformer model.
        to_logits (nn.Linear): Linear layer to convert embeddings to logits.
    """

    def __init__(
        self,
        dim,
        num_tokens,
        max_seq_len,
        action_dim=None,
        is_guided=True,
        gradient_shrink_alpha=0.1,
        heads=8,
        dim_head=64,
        attn_dropout=0.0,
        ff_dropout=0.0,
        image_size=128,
        patch_size=8,
        num_blocks=8,
        use_token=True,
        *args,
        **kwargs,
    ):
        """
        Initialize the MaskGit model.

        Args:
            dim (int): The dimension of the model.
            num_tokens (int): The number of tokens in the vocabulary.
            max_seq_len (int): The maximum sequence length.
            action_dim (int): The dimension of the action tokens.
            is_guided (bool): Whether the model is guided.
            gradient_shrink_alpha (float): Alpha value for gradient shrinking.
            heads (int): The number of attention heads.
            dim_head (int): The dimension of each attention head.
            attn_dropout (float): Dropout rate for attention.
            ff_dropout (float): Dropout rate for feedforward layers.
            image_size (int): The size of the input images.
            patch_size (int): The size of the patches.
            num_blocks (int): The number of transformer blocks.
            **kwargs: Additional keyword arguments.
        """

        assert not is_guided or exists(action_dim), (
            "Action dim must be provided for guided dynamics"
        )

        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.action_dim_effective = action_dim
        self.use_token = use_token

        if not is_guided:
            self.action_dim_effective = 0

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)

        self.mask_id = num_tokens
        self.is_guided = is_guided

        # learnable token for the masked token
        self.token_emb = nn.Embedding(
            num_tokens + 1, dim
        )  # last token is used as mask_id

        self.mask_token_template = torch.zeros(
            (self.dim)
        )

        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.gradient_shrink_alpha = gradient_shrink_alpha  # used with great success in cogview and GLM 130B attention net

        self.continuous_pos_bias = ContinuousPositionBias(
            dim=dim_head, heads=heads, num_dims=2
        )

        transformer_kwargs = dict(
            dim=dim + self.action_dim_effective,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=True,
            peg=True,
            peg_causal=True,
            num_blocks=num_blocks,
            **kwargs,
        )

        self.transformer = STTransformer(order="st", **transformer_kwargs)

        self.to_logits = nn.Linear(dim + self.action_dim_effective, num_tokens)

        # wandb config
        config = {}
        arguments = locals()
        for key in arguments.keys():
            if key not in ["self", "config", "__class__"]:
                config[key] = arguments[key]
        self.config = config

    @property
    def patch_height_width(self):
        """

        Calculate the height and width of patches.

        Returns:
            tuple: A tuple containing the patch height and width.
        """
        return (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )

    def forward(
        self,
        video_tokens_ids,
        actions,
        prompt_tokens=None,
        video_mask=None,
        video_patch_shape=None,
        return_embeds=False,
        tokenizer=None,
        **kwargs,
    ):
        """
        Forward pass of the MaskGit model.

        Args:
            video_tokens_ids (torch.Tensor): The input video token IDs.
            actions (torch.Tensor): The input action tokens.
            video_mask (torch.Tensor, optional): Mask for the video tokens.
            video_patch_shape (tuple, optional): The shape of the video patches.
            return_embeds (bool): Whether to return embeddings instead of logits.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output logits or embeddings.

        Raises:
            AssertionError: If the input shape is invalid or sequence length exceeds the maximum.
        """
        assert video_tokens_ids.ndim in {
            2,
            4,
        }, (
            "video token ids must be of shape (batch, seq) or (batch, frame, height, width)"
        )

        if video_tokens_ids.ndim == 4:
            video_patch_shape = video_tokens_ids.shape[1:]
            video_tokens_ids = rearrange(video_tokens_ids, "b ... -> b (...)")

        b, n, device = *video_tokens_ids.shape, video_tokens_ids.device

        assert exists(video_patch_shape), "video patch shape must be given"

        t, h, w = video_patch_shape

        rel_pos_bias = self.continuous_pos_bias(h, w, device=device)

        video_shape = (b, t, h, w)
        mask = video_tokens_ids == self.mask_id

        if self.use_token:
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided when use_token is True")

            with torch.no_grad():
                video_tokens_ids_zeroed = torch.where(mask, 0, video_tokens_ids)
                codes = tokenizer.vq.codebook[video_tokens_ids_zeroed]
                codes = tokenizer.vq.project_out(codes)

            mask_token_template = self.mask_token_template.to(device).type_as(codes)
            video_tokens = codes.clone() * 10
            video_tokens[mask] = mask_token_template.unsqueeze(0)

            video_tokens = self.token_emb(video_tokens_ids) + video_tokens
        else:
            video_tokens = self.token_emb(video_tokens_ids)

        assert n <= self.max_seq_len, (
            f"the video token sequence length you are passing in ({n}) is greater than the `max_seq_len` ({self.max_seq_len}) set on your `MaskGit`"
        )
        video_tokens = self.pos_emb(torch.arange(n, device=device)) + video_tokens

        video_tokens = (
            video_tokens * self.gradient_shrink_alpha
            + video_tokens.detach() * (1 - self.gradient_shrink_alpha)
        )

        if video_tokens.ndim == 3:
            video_tokens = rearrange(video_tokens, "b (t h w) d -> b t h w d", h=h, w=w)
        if exists(video_mask):
            video_mask = rearrange(video_mask, "b (t h w) -> b t h w", h=h, w=w)

        # concatenate the action tokens to the video tokens
        if self.is_guided:
            actions = actions.unsqueeze(2).unsqueeze(2).expand(-1, -1, h, w, -1)
            video_action_tokens = torch.cat([video_tokens, actions], dim=-1)
        else:
            actions = actions.expand(-1, -1, h, w, -1)
            video_action_tokens = video_tokens + actions

        pattern = "b t h w d"
        spatial_pattern = "(b t) (h w) d"
        temporal_pattern = "(b h w) t d"

        video_tokens = self.transformer(
            video_action_tokens,
            pattern=pattern,
            spatial_pattern=spatial_pattern,
            temporal_pattern=temporal_pattern,
            video_shape=video_shape,
            attn_bias=rel_pos_bias,
            self_attn_mask=video_mask,
            **kwargs,
        )

        video_tokens = rearrange(
            video_tokens,
            f"{pattern} -> b (t h w) d",
            b=b,
            h=h,
            w=w,
        )

        if return_embeds:
            return video_tokens

        return self.to_logits(video_tokens)


class Dynamics(nn.Module):
    """
    Dynamics module for handling the dynamics of the Genie model.

    Args:
        maskgit (GenieMaskGit): The GenieMaskGit model for token prediction.
        sample_temperature (float): Temperature for sampling.
    """

    def __init__(
        self,
        *,
        maskgit: MaskGIT,
        tokenizer_codebook=None,
        inference_steps=1,
        sample_temperature=1.0,
        mask_schedule="cosine",
        use_cross_entropy_loss=True,
        use_focal_loss=False,
        use_distance_weighted_loss=False,
    ):
        assert use_cross_entropy_loss != use_focal_loss, (
            "Only one CE loss function can be used at a time"
        )

        super().__init__()

        self.maskgit = maskgit
        self.mask_id = maskgit.mask_id
        self.inference_steps = (inference_steps,)
        self.sample_temperature = (sample_temperature,)
        self.mask_schedule = (mask_schedule,)
        self.use_cross_entropy_loss = use_cross_entropy_loss
        self.use_focal_loss = use_focal_loss
        self.use_distance_weighted_loss = use_distance_weighted_loss

        self.focal_loss = FocalLoss(gamma=2.0)

        if use_distance_weighted_loss:
            self.distance_weighted_loss = DistanceWeightedCrossEntropyLoss(
                tokenizer_codebook, _lamda=3
            )

    def loss_function(self, logits, target_indices, accelerator_tracker_dict=None):
        loss = 0
        use_loss_tracking = accelerator_tracker_dict is not None

        if use_loss_tracking:
            tracker = accelerator_tracker_dict["tracker"]
            step = accelerator_tracker_dict["step"]
            use_loss_tracking = step % 50 == 0

        if self.use_cross_entropy_loss:
            ce_loss = F.cross_entropy(logits, target_indices)
            if use_loss_tracking:
                tracker.log(
                    {
                        "Train Cross Entropy Loss": ce_loss.item(),
                    },
                    step=step,
                )
            loss += ce_loss

        if self.use_focal_loss:
            focal_loss = self.focal_loss(logits, target_indices, alpha=2)
            if use_loss_tracking:
                tracker.log(
                    {
                        "Train Focal Loss": focal_loss.item(),
                    },
                    step=step,
                )
            loss += focal_loss

        if self.use_distance_weighted_loss:
            dw_loss = self.distance_weighted_loss(logits, target_indices)
            if use_loss_tracking:
                tracker.log(
                    {
                        "Train Distance Weighted Loss": dw_loss.item(),
                    },
                    step=step,
                )
            loss += dw_loss

        return loss

    def accelator_log(self, accelerator_tracker_dict, loss):
        """
        Log the loss values using the accelerator tracker.

        Args:
            accelerator_tracker_dict (dict): Dictionary containing tracker information.
            loss (torch.Tensor): The main loss value.
        """
        if exists(accelerator_tracker_dict):
            train = accelerator_tracker_dict["train"]
            tracker = accelerator_tracker_dict["tracker"]
            step = accelerator_tracker_dict["step"]

            mode = "Train" if train else "Validation"
            tracker.log(
                {
                    f"{mode} Total Loss": loss.item(),
                },
                step=step,
            )

    def compute_confidence_scores(
        self,
        pred_video_ids,
        logits,
        mask,
    ):
        """
        Compute confidence scores for predicted tokens.

        Args:
            patch_shape (tuple): Shape of the video patch.
            video_token_ids (torch.Tensor): Video token IDs.
            actions (torch.Tensor): Action tensors.
            pred_video_ids (torch.Tensor): Predicted video token IDs.
            logits (torch.Tensor): Logits from the model.
            mask (torch.Tensor): Mask tensor.
            prime_token_ids (torch.Tensor, optional): Prime token IDs.

        Returns:
            torch.Tensor: Confidence scores.
        """
        probs = logits.softmax(dim=-1)
        scores = probs.gather(2, rearrange(pred_video_ids, "... -> ... 1"))
        scores = 1 - rearrange(scores, "... 1 -> ...")
        scores = torch.where(mask, scores, -1e4)

        return scores

    @eval_decorator
    @torch.no_grad()
    def sample(
        self,
        prime_token_ids=None,  # B x C x T x H x W
        actions=None,
        prompt_tokens=None,
        num_tokens=None,
        patch_shape=None,
        inference_steps=None,
        mask_schedule=None,
        sample_temperature=None,
        return_confidence=False,
        return_logits=False,
        tokenizer=None,
        *args,
        **kwargs,
    ):
        """
        Sample video frames based on actions and prime frames.

        Args:
            num_frames (int): Number of frames to generate.
            actions (torch.Tensor): Action tensors.
            prime_frames (torch.Tensor): Prime frames to start generation from.

        Returns:
            torch.Tensor: Generated video frames.
        """

        if not exists(inference_steps):
            inference_steps = self.inference_steps
        if not exists(sample_temperature):
            sample_temperature = self.sample_temperature
        if not exists(mask_schedule):
            mask_schedule = self.mask_schedule

        assert exists(prime_token_ids), "prime_token_ids must be provided"
        assert exists(actions), "actions must be provided"
        assert exists(num_tokens), "num_tokens must be provided"
        assert exists(patch_shape), "patch_shape must be provided"

        device = next(self.parameters()).device

        # get video token ids
        batch_size = actions.shape[0]
        shape = (batch_size, num_tokens)

        t, h, w = patch_shape

        video_token_ids = torch.full(shape, self.mask_id, device=device)
        mask = torch.ones(shape, device=device, dtype=torch.bool)

        scores = None
        all_scores = []
        for step in range(inference_steps):
            is_first_step = step == 0
            is_last_step = step == inference_steps - 1
            steps_til_x0 = inference_steps - (step + 1)

            if not is_first_step:
                ratio = torch.full((1,), (step + 1) / inference_steps, device=device)
                mask_ratio = mask_ratio_schedule(ratio, num_tokens, mask_schedule)
                num_tokens_mask = int(mask_ratio * num_tokens)
                _, indices = scores.topk(num_tokens_mask, dim=-1)
                mask = torch.zeros(shape, device=device).scatter(1, indices, 1).bool()

            video_token_ids = torch.where(mask, self.mask_id, video_token_ids)

            input_token_ids = torch.cat((prime_token_ids, video_token_ids), dim=-1)

            input_token_ids = rearrange(
                input_token_ids, "b (t h w) -> b t h w", h=h, w=w
            )
            input_token_ids = input_token_ids[:, :-1]

            logits = self.maskgit.forward(
                input_token_ids,
                actions,
                prompt_tokens=prompt_tokens,
                video_patch_shape=patch_shape,
                tokenizer=tokenizer
            )[:, -num_tokens:]

            temperature = sample_temperature * (steps_til_x0 / inference_steps)
            pred_video_ids = gumbel_sample(logits, temperature=temperature)

            video_token_ids = (
                pred_video_ids  # torch.where(mask, pred_video_ids, video_token_ids)
            )

            if not is_last_step or return_confidence:
                scores = self.compute_confidence_scores(
                    pred_video_ids,
                    logits,
                    mask,
                )[:, -num_tokens:]

                if return_confidence:
                    all_scores.append(scores)

        if return_logits:
            return logits

        if return_confidence:
            return video_token_ids, all_scores

        return video_token_ids

    def forward(
        self,
        video_codebook_ids=None,
        actions=None,
        video_mask=None,
        accelerator_tracker_dict=None,
        return_token_ids=False,
        return_logits=False,
        tokenizer=None,
        *args,
        **kwargs,
    ):
        """
        Forward pass of the Dynamics module.

        Args:
            video_codebook_ids (torch.Tensor): Video codebook IDs.
            latent_actions (torch.Tensor): Latent action representations.
            video_mask (torch.Tensor, optional): Mask for valid video tokens.
            accelerator_tracker_dict (dict, optional): Dictionary for logging.
            return_token_ids (bool): Whether to return token IDs.

        Returns:
            torch.Tensor or tuple: Loss value or (loss, token_ids) if return_token_ids is True.
        """

        assert exists(video_codebook_ids) and exists(actions), (
            "video_codebook_ids and actions must be provided"
        )

        # Prepare input for the model
        maskgit_input_video_codebook_ids = video_codebook_ids[:, :-1]
        video_codebook_ids = video_codebook_ids[:, 1:]

        video_codebook_ids, packed_shape = pack([video_codebook_ids], "b *")
        maskgit_input_video_codebook_ids, _ = pack(
            [maskgit_input_video_codebook_ids], "b *"
        )

        batch, seq, device = (
            *maskgit_input_video_codebook_ids.shape,
            maskgit_input_video_codebook_ids.device,
        )

        if not exists(video_mask):
            video_mask = torch.ones((batch, seq), device=device).bool()

        # Apply masking
        mask_token_prob = (torch.rand(1) * 0.5 + 0.5).item()
        token_mask = apply_bernoulli_mask(video_mask, mask_token_prob)

        masked_input_video = torch.where(
            token_mask, self.mask_id, maskgit_input_video_codebook_ids
        )

        (masked_input_video,) = unpack(masked_input_video, packed_shape, "b *")

        # Forward pass through MaskGit

        logits = self.maskgit(
            masked_input_video,
            actions,
            video_mask=video_mask,
            tokenizer=tokenizer,
        )

        # Compute loss
        loss = self.loss_function(
            logits[token_mask],
            video_codebook_ids[token_mask],
            accelerator_tracker_dict,
        )

        if return_logits:
            return loss, logits

        if return_token_ids:
            pred_video_ids = gumbel_sample(
                logits, temperature=self.critic_train_sample_temperature
            )
            returned_token_ids = torch.where(
                token_mask, pred_video_ids, video_codebook_ids
            )
            return loss, returned_token_ids

        return loss
