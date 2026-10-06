import os
import torch

from models import (
    Dynamics,
    GenieRedux,
    GenieReduxGuided,
    LatentActionModel,
    MaskGIT,
    Tokenizer,
)


def construct_model(config):
    if config.model not in ["tokenizer", "genie_redux", "genie_redux_guided", "genie_redux_guided_pretrain"]:
        raise ValueError(f"Unknown model: {config.model}")

    tokenizer = Tokenizer(
        dim=config.tokenizer.dim,
        codebook_size=config.tokenizer.codebook_size,
        image_size=config.tokenizer.image_size,
        patch_size=config.tokenizer.patch_size,
        wandb_mode=config.train.wandb_mode,
        temporal_patch_size=config.tokenizer.temporal_patch_size,  # temporal patch size
        num_blocks=config.tokenizer.num_blocks,  # nb of blocks in st transformer
        dim_head=config.tokenizer.dim_head,  # hidden size in transfo
        heads=config.tokenizer.heads,  # nb of heads for multi head transfo
        ff_mult=config.tokenizer.ff_mult,  # 32 * 64 = 2048 MLP size in transfo out
        vq_loss_w=config.tokenizer.vq_loss_weight,  # commit loss weight
        recon_loss_w=config.tokenizer.recons_loss_weight,  # reconstruction loss weight
    )

    if config.model == "tokenizer":
        return tokenizer

    # Tokenizer weights are now loaded in the training script (train_genie_redux.py)
    # to enforce clear precedence and error handling across model/tokenizer options.

    # Determine guidance from config instead of model name
    is_guided = getattr(config.dynamics, "is_guided", False)

    # If guided and using action embeddings, actions are embedded to the same
    # dimensionality as video tokens (config.dynamics.dim). In that case the
    # transformer must be instantiated with an effective action dim equal to
    # `dim`, not the discrete action count.
    effective_action_dim = (
        config.dynamics.dim
        if is_guided and getattr(config.dynamics, "use_action_embeddings", False)
        else config.dynamics.action_dim
    )

    maskgit = MaskGIT(
        dim=config.dynamics.dim,
        is_guided=is_guided,
        action_dim=effective_action_dim,
        num_tokens=config.tokenizer.codebook_size,
        heads=config.dynamics.heads,
        dim_head=config.dynamics.dim_head,
        num_blocks=config.dynamics.num_blocks,
        max_seq_len=config.dynamics.max_seq_len,
        image_size=config.dynamics.image_size,
        patch_size=config.dynamics.patch_size,
        use_token=config.dynamics.use_token,
    )

    dynamics = Dynamics(
        maskgit=maskgit,
        inference_steps=1,
        sample_temperature=config.dynamics.sample_temperature,
        mask_schedule="cosine",
    )

    if is_guided:
        # optionally enable action embeddings if configured
        use_action_embeddings = config.dynamics.use_action_embeddings
        model = GenieReduxGuided(
            tokenizer, dynamics, use_action_embeddings=use_action_embeddings
        )
    else:
        latent_action_model = LatentActionModel(
            dim=config.lam.dim,
            codebook_size=config.lam.codebook_size,
            image_size=config.lam.image_size,
            patch_size=config.lam.patch_size,
            wandb_mode=config.train.wandb_mode,
            temporal_patch_size=config.lam.temporal_patch_size,  # temporal patch size
            num_blocks=config.lam.num_blocks,  # nb of blocks in st transformer
            dim_head=config.lam.dim_head,  # hidden size in transfo
            heads=config.lam.heads,  # nb of heads for multi head transfo
            ff_mult=config.lam.ff_mult,  # 32 * 64 = 2048 MLP size in transfo out
            vq_loss_w=config.lam.vq_loss_weight,  # commit loss weight
            recon_loss_w=config.lam.recons_loss_weight,  # reconstruction loss weight
        )

        model = GenieRedux(tokenizer, latent_action_model, dynamics)

    return model
