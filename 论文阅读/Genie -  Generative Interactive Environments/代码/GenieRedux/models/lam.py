import torch
import torch.nn.functional as F
from torch import nn


from einops import repeat, pack
from einops.layers.torch import Rearrange

from models.components import STViViT

# helpers


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


class LatentActionModel(STViViT):
    """
    LatentActionModel class for learning latent representations of actions in video sequences.
    """

    """
    einstein notations:

    b - batch
    c - channels
    t - time
    d - feature dimension
    p1, p2, pt - image patch sizes and then temporal patch size
    """

    def __init__(
        self,
        *,
        dim,
        codebook_size,
        image_size,
        patch_size,
        temporal_patch_size,
        num_blocks,
        wandb_mode="disabled",
        codebook_dim=32,
        dim_head=64,
        heads=8,
        channels=3,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4.0,
        recon_loss_w=1.0,
        vq_loss_w=1.0,
    ):
        """
        Initializes the LatentActionModel.

        Args:
            dim (int): The feature dimension.
            codebook_size (int): The size of the VQ codebook.
            image_size (int): The height and width of the input images.
            patch_size (int): The size of the patches extracted from the images.
            temporal_patch_size (int): The size of temporal patches.
            num_blocks (int): The number of transformer blocks.
            wandb_mode (str): Mode for logging to Weights and Biases (default: disabled).
            codebook_dim (int): The dimension of the codebook (default: 32).
            dim_head (int): The dimension of the attention heads (default: 64).
            heads (int): The number of attention heads (default: 8).
            channels (int): The number of input image channels (default: 3).
            attn_dropout (float): Dropout for attention (default: 0.0).
            ff_dropout (float): Dropout for feed-forward layers (default: 0.0).
            ff_mult (float): Multiplication factor for the feed-forward layer size (default: 4.0).
            recon_loss_w (float): Weight for the reconstruction loss (default: 1.0).
            vq_loss_w (float): Weight for the VQ loss (default: 1.0).

        """

        super().__init__(
            dim=dim,
            codebook_size=codebook_size,
            image_size=image_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            num_blocks=num_blocks,
            wandb_mode=wandb_mode,
            codebook_dim=codebook_dim,
            dim_head=dim_head,
            heads=heads,
            channels=channels,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult=ff_mult,
            vq_loss_w=vq_loss_w,
            recon_loss_w=recon_loss_w,
        )

        h, w = self.patch_height_width

        # Layer to embed actions
        self.to_action_emb = nn.Sequential(
            Rearrange("b t ... -> b t (...)"),
            nn.Linear(h * w * dim, dim),
            nn.LayerNorm(dim),
        )

    def accelator_log(self, accelerator_tracker_dict, vq_loss, recon_loss):
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
            tracker.log({f"{mode} LAM VQ loss": vq_loss.item()}, step=step)
            tracker.log({f"{mode} LAM recon loss": recon_loss.item()}, step=step)

    def get_codes_from_indices(self, indices):
        """
        Retrieves the latent codes corresponding to the given codebook indices.

        Args:
            indices (torch.Tensor): The codebook indices.

        Returns:
            torch.Tensor: The corresponding latent codes from the codebook.
        """

        codes = self.vq.codebook[indices]
        projected_out_codes = self.vq.project_out(codes)
        return projected_out_codes

    def forward(
        self,
        videos,
        action_ids=None,
        mask=None,
        return_tokens_only=False,
        return_tokens=False,
        return_recons=False,
        return_recons_only=False,
        return_only_codebook_ids=False,
        accelerator_tracker_dict=None,
        *args,
        **kwargs,
    ):
        """
        Forward pass of the model for encoding and decoding video frames.

        Args:
            videos (torch.Tensor): Input video tensor of shape (batch, channels, frames, height, width).
            action_ids (torch.Tensor, optional): Action tokens to use for decoding (default: None).
            mask (torch.Tensor, optional): Mask for the input frames (default: None).
            return_tokens_only (bool, optional): If True, return only the encoded tokens (default: False).
            return_recons (bool, optional): If True, return both the loss and reconstructed video (default: False).
            return_recons_only (bool, optional): If True, return only the reconstructed video (default: False).
            return_only_codebook_ids (bool, optional): If True, return only the codebook indices (default: False).
            accelerator_tracker (optional): Used for logging with wandb (default: None).
            step (int, optional): Training step (default: 0).
            log_every (int, optional): Frequency of logging (default: 50).

        Returns:
            torch.Tensor: Depending on the arguments, returns either the loss, tokens, or reconstructed video.
        """

        # 4 is BxCxHxW (for images), 5 is BxCxFxHxW
        assert videos.ndim == 5

        b, c, f, *image_dims, device = *videos.shape, videos.device

        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f
        assert divisible_by(
            f - 1, self.temporal_patch_size
        ), f"number of frames ({f}) minus one ({f - 1}) must be divisible by temporal patch size ({self.temporal_patch_size})"

        first_frame, rest_frames = videos[:, :, :1], videos[:, :, 1:]

        # derive patches
        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb(rest_frames)

        # simple cat, normal
        enc_input_tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)

        # save height and width in
        shape = enc_input_tokens.shape
        *_, h, w, _ = shape

        # encode - spatial-temporal
        # encode_input_tokens shape: (b, t, h, w, d)
        tokens = self.encode(enc_input_tokens)
        tokens = self.to_action_emb(tokens)

        tokens = tokens[:, 1:]

        # quantize
        tokens, packed_thw_shape = pack([tokens], "b * d")

        tokens, indices, vq_loss = self.vq(tokens, mask=None)

        if return_only_codebook_ids:
            return indices

        if exists(action_ids):
            action_tokens = self.vq.codebook[action_ids]
            tokens = self.vq.project_out(action_tokens)

        # now repeate over the spatial dimensions
        tokens = repeat(tokens, "b t d -> b t h w d", h=h, w=w)

        if return_tokens_only:
            return tokens

        dec_input_tokens = enc_input_tokens[:, :-1] + tokens

        recon_video = self.decode(dec_input_tokens)

        if return_recons_only:
            return recon_video

        # LOSS COMPUTATION

        if exists(mask):
            # variable lengthed video / images training
            recon_loss = F.mse_loss(videos[:, :, 1:], recon_video, reduction="none")
            recon_loss = recon_loss[repeat(mask, "b t -> b c t", c=c)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.mse_loss(videos[:, :, 1:], recon_video)

        # combine losses

        loss = self.vq_loss_w * vq_loss + self.recon_loss_w * recon_loss

        # check if the model is in training mode
        self.accelator_log(accelerator_tracker_dict, vq_loss, recon_loss)

        if return_tokens:
            return loss, tokens

        if return_recons:
            return loss, recon_video

        return loss

    # generate random action indices for each of the indices given, but different from those indices themselves. the generated indices shoud be from 0 to codebook_size - 1

    def generate_random_different_actions(self, actions_indices, device):
        codebook_size = self.codebook_size
        shape = actions_indices.shape
        random_actions = torch.randint(0, codebook_size, shape, device=device)

        while torch.any(random_actions == actions_indices):
            random_actions = torch.where(
                random_actions == actions_indices,
                torch.randint(0, codebook_size, shape, device=device),
                random_actions,
            )

        return random_actions

    def lam_vs_random_actions(self, video, random_rollout_steps: list[int] = None):
        assert video.ndim == 5

        b, c, f, *image_dims, device = *video.shape, video.device

        assert not exists(random_rollout_steps) or max(random_rollout_steps) < f
        if not exists(random_rollout_steps):
            random_rollout_steps = [f - 1]

        assert tuple(image_dims) == self.image_size
        assert divisible_by(
            f - 1, self.temporal_patch_size
        ), f"number of frames ({f}) minus one ({f - 1}) must be divisible by temporal patch size ({self.temporal_patch_size})"

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        # derive patches
        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb(rest_frames)

        # simple cat, normal
        enc_input_tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)

        # save height and width in
        shape = enc_input_tokens.shape
        *_, h, w, _ = shape

        tokens = self.encode(enc_input_tokens)
        tokens = self.to_action_emb(tokens)

        tokens = tokens[:, 1:]

        # quantize
        tokens, packed_thw_shape = pack([tokens], "b * d")

        tokens, action_indices, vq_loss = self.vq(tokens, mask=None)

        # now repeate over the spatial dimensions
        tokens = repeat(tokens, "b t d -> b t h w d", h=h, w=w)

        dec_input_tokens = enc_input_tokens[:, :-1] + tokens

        lam_recon_video = self.decode(dec_input_tokens)

        random_rollout_recon_videos = []
        random_actions_indices = self.generate_random_different_actions(
            action_indices, device
        )
        for rollout_step in random_rollout_steps:
            actions_to_take = action_indices.clone()[:, :rollout_step]
            actions_to_take[:, rollout_step - 1] = random_actions_indices[
                :, rollout_step - 1
            ]
            frame_embedds = enc_input_tokens[:, :rollout_step]

            action_tokens = self.vq.codebook[actions_to_take]
            action_tokens = self.vq.project_out(action_tokens)
            action_tokens = repeat(action_tokens, "b t d -> b t h w d", h=h, w=w)

            dec_input_tokens = frame_embedds + action_tokens

            random_rollout_recon_video = self.decode(dec_input_tokens)

            random_rollout_recon_videos.append(random_rollout_recon_video)

        return lam_recon_video, random_rollout_recon_videos
