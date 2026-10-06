import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from torch import nn

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


def log_wandb_all_losses(
    accelerator,
    vq_loss,
    recon_loss,
    step,
    is_training=True,
):
    """
    Logs various losses to Weights & Biases (wandb).

    Args:
        accelerator: The Hugging Face Accelerator object.
        vq_loss (torch.Tensor): The Vector Quantization loss.
        recon_loss (torch.Tensor): The reconstruction loss.
        step (int): The current training step.
        is_training (bool): Whether the model is in training mode. Default is True.

    Returns:
        None
    """
    if not exists(accelerator) or not accelerator.is_main_process:
        return
    mode = "Train" if is_training else "Validation"
    accelerator.log({f"{mode} VQ loss": vq_loss.item()}, step=step)
    accelerator.log({f"{mode} reconstruction loss": recon_loss.item()}, step=step)

    return


class Tokenizer(STViViT):
    """
    A neural network module for tokenizing video data.

    This class implements a tokenizer that can encode video frames into discrete tokens
    and decode them back to video frames. It uses a combination of spatial and temporal
    transformers along with vector quantization.

    Attributes:
        wandb_mode (str): The mode for Weights & Biases logging.
        force_cpu (bool): Whether to force CPU usage instead of GPU.
        vq_loss_w (float): The weight for the Vector Quantization loss.
        recon_loss_w (float): The weight for the reconstruction loss.
        gp_weight (float): The weight for gradient penalty.
        image_size (tuple): The size of the input images (height, width).
        patch_size (tuple): The size of the patches (height, width).
        temporal_patch_size (int): The size of temporal patches.
        spatial_rel_pos_bias (ContinuousPositionBias): The spatial relative position bias.
        to_patch_emb_first_frame (nn.Sequential): Embedding layer for the first frame.
        to_patch_emb (nn.Sequential): Embedding layer for the rest of the frames.
        encoder (STTransformer): The encoder transformer.
        vq (VectorQuantize): The Vector Quantization layer.
        decoder (STTransformer): The decoder transformer.
        to_pixels_first_frame (nn.Sequential): Layer to convert tokens to pixels for the first frame.
        to_pixels (nn.Sequential): Layer to convert tokens to pixels for the rest of the frames.
        config (dict): Configuration dictionary containing all initialization parameters.
    """

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
        train_decoder_only=False,
    ):
        """
        Initializes the Tokenizer.

        Args:
            dim (int): The dimension of the model. Default is 512.
            codebook_size (int): The size of the codebook for vector quantization. Default is 1024.
            image_size (int or tuple): The size of the input images. Default is 64.
            patch_size (int or tuple): The size of the patches. Default is 4.
            temporal_patch_size (int): The size of temporal patches. Default is 1.
            num_blocks (int): The number of transformer blocks. Default is 8.
            wandb_mode (str): The mode for Weights & Biases logging. Default is "disabled".
            codebook_dim (int): The dimension of the codebook. Default is 32.
            dim_head (int): The dimension of each attention head. Default is 64.
            heads (int): The number of attention heads. Default is 8.
            channels (int): The number of input channels. Default is 3.
            attn_dropout (float): The dropout rate for attention layers. Default is 0.0.
            ff_dropout (float): The dropout rate for feedforward layers. Default is 0.0.
            ff_mult (float): The multiplier for the feedforward dimension. Default is 4.0.
            vq_loss_w (float): The weight for the Vector Quantization loss. Default is 1.0.
            recon_loss_w (float): The weight for the reconstruction loss. Default is 1.0.
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
            enable_decoder=enable_decoder,
        )

        self.train_decoder_only = train_decoder_only

    def calculate_video_token_mask(self, videos, video_frame_mask):
        """
        Calculate a mask for video tokens based on the input video and frame mask.

        Args:
            videos (torch.Tensor): Input video tensor.
            video_frame_mask (torch.Tensor): Mask indicating valid frames in the video.

        Returns:
            torch.Tensor: Mask for video tokens.
        """
        *_, h, w = videos.shape
        ph, pw = self.patch_size

        # Ensure the number of frames (minus the first frame) is divisible by temporal patch size
        assert torch.all(
            ((video_frame_mask.sum(dim=-1) - 1) % self.temporal_patch_size) == 0
        ), (
            "number of frames must be divisible by temporal patch size, subtracting off the first frame"
        )

        # Split mask into first frame and rest of the frames
        first_frame_mask, rest_frame_mask = (
            video_frame_mask[:, :1],
            video_frame_mask[:, 1:],
        )

        # Reshape rest frame mask to group by temporal patch size
        rest_vq_mask = rearrange(
            rest_frame_mask, "b (f p) -> b f p", p=self.temporal_patch_size
        )

        # Combine first frame mask with the rest, considering any valid frame in temporal patch
        video_mask = torch.cat((first_frame_mask, rest_vq_mask.any(dim=-1)), dim=-1)

        # Repeat mask for each spatial patch
        return repeat(video_mask, "b f -> b (f hw)", hw=(h // ph) * (w // pw))

    def get_video_patch_shape(self, num_frames, num_first_frames=1):
        """
        Calculate the shape of video patches.

        Args:
            num_frames (int): Total number of frames in the video.
            num_first_frames (int): Number of frames to be treated separately (default: 1).

        Returns:
            tuple: Shape of video patches (frames, height, width).
        """
        patch_frames = 0

        # Handle first frames separately
        if num_first_frames > 0:
            num_frames -= num_first_frames
            patch_frames += num_first_frames

        # Calculate remaining patch frames
        patch_frames += num_frames // self.temporal_patch_size

        return (patch_frames, *self.patch_height_width)

    @property
    def image_num_tokens(self):
        """
        Calculate the number of tokens in a single image.

        Returns:
            int: Number of tokens in an image.
        """
        return int(self.image_size[0] / self.patch_size[0]) * int(
            self.image_size[1] / self.patch_size[1]
        )

    def frames_per_num_tokens(self, num_tokens):
        """
        Calculate the number of frames represented by a given number of tokens.

        Args:
            num_tokens (int): Number of tokens.

        Returns:
            int: Number of frames represented by the tokens.
        """
        tokens_per_frame = self.image_num_tokens

        assert (num_tokens % tokens_per_frame) == 0, (
            f"number of tokens must be divisible by number of tokens per frame {tokens_per_frame}"
        )
        assert num_tokens > 0

        pseudo_frames = num_tokens // tokens_per_frame
        return (pseudo_frames - 1) * self.temporal_patch_size + 1

    def num_tokens_per_frames(self, num_frames, num_first_frames=1):
        """
        Calculate the number of tokens needed to represent a given number of frames.

        Args:
            num_frames (int): Number of frames.
            num_first_frames (int): Number of frames to be treated separately (default: 1).

        Returns:
            int: Number of tokens needed to represent the frames.
        """
        image_num_tokens = self.image_num_tokens

        total_tokens = 0

        # Handle first frames separately
        if num_first_frames > 0:
            num_frames -= num_first_frames
            total_tokens += num_first_frames * image_num_tokens

        assert (num_frames % self.temporal_patch_size) == 0

        # Calculate tokens for remaining frames
        return (
            total_tokens + int(num_frames / self.temporal_patch_size) * image_num_tokens
        )

    def forward(
        self,
        videos=None,
        mask=None,
        return_recons=False,
        return_recons_only=False,
        return_only_codebook_ids=False,
        accelerator_tracker=None,
        step=0,
        log_every=50,
        **kwargs,
    ):
        """
        Forward pass of the Tokenizer model.

        Args:
            videos (torch.Tensor): Input video tensor.
            mask (torch.Tensor): Mask for variable-length videos.
            return_recons (bool): Whether to return reconstructions.
            return_recons_only (bool): Whether to return only reconstructions.
            return_only_codebook_ids (bool): Whether to return only codebook indices.
            accelerator_tracker: Accelerator for logging.
            step (int): Current step for logging.
            log_every (int): Frequency of logging.

        Returns:
            torch.Tensor or tuple: Loss or tuple of (loss, reconstructions).
        """
        assert videos is not None, "video must be provided"
        assert videos.ndim == 5

        b, c, f, *image_dims, device = *videos.shape, videos.device

        # Validate input dimensions
        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f
        assert divisible_by(f - 1, self.temporal_patch_size), (
            f"number of frames ({f}) minus one ({f - 1}) must be divisible by temporal patch size ({self.temporal_patch_size})"
        )

        # Split video into first frame and rest frames
        first_frame, rest_frames = videos[:, :, :1], videos[:, :, 1:]

        # Embed patches
        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb(rest_frames)

        # Concatenate tokens
        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)

        shape = tokens.shape
        *_, h, w, _ = shape

        # Encode tokens
        tokens = self.encode(tokens)

        # Quantize
        tokens, packed_fhw_shape = pack([tokens], "b * d")

        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(videos, mask)
        tokens, indices, vq_loss = self.vq(tokens, mask=vq_mask)

        if self.train_decoder_only:
            tokens = tokens.detach()

        if return_only_codebook_ids:
            (indices,) = unpack(indices, packed_fhw_shape, "b *")
            return indices

        tokens = rearrange(tokens, "b (t h w) d -> b t h w d", h=h, w=w)

        # Decode tokens
        recon_video = self.decode(tokens)

        if return_recons_only:
            returned_recon = recon_video
            return returned_recon

        # Compute losses
        if exists(mask):
            # Variable-length video / images training
            recon_loss = F.mse_loss(videos, recon_video, reduction="none")
            recon_loss = recon_loss[repeat(mask, "b t -> b c t", c=c)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.mse_loss(videos, recon_video)

        # Combine losses
        if self.train_decoder_only:
            loss = self.recon_loss_w * recon_loss
        else:
            loss = self.vq_loss_w * vq_loss + self.recon_loss_w * recon_loss

        # Log losses if needed
        if self.wandb_mode != "disabled" and step % log_every == 0:
            log_wandb_all_losses(
                accelerator_tracker,
                vq_loss,
                recon_loss,
                step,
                self.training,
            )

        if return_recons:
            returned_recon = recon_video
            return loss, returned_recon

        return loss
