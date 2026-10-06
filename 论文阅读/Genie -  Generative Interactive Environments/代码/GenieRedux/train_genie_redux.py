import math
import os

# set the current working directory as the project root directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from prettytable import PrettyTable
from torch.utils.data import Subset

from data.data import (
    DatasetOutputFormat,
    MultiEnvironmentDataset,
    TransformsGenerator,
)
from models import construct_model
from training import Trainer
from utils.utils import debug
from omegaconf import OmegaConf


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Number of params in total: ", pytorch_total_params)
    return total_params


def run(args):
    dataset_folder = f"{args.train.dataset_root_dpath}/{args.train.dataset_name}"
    # cache_dpath = (
    #     args.train.wandb_dpath if args.train.wandb_dpath != "./wandb" else "./cache"
    # )

    model = construct_model(args)
    num_frames = args.train.num_frames
    n_workers = args.train.n_data_workers
    n_total_samples = args.train.n_total_samples
    n_envs = args.train.n_envs

    # Enforce explicit and predictable loading behavior for model/tokenizer
    if getattr(args, "model", None) != "tokenizer":
        have_model = bool(getattr(args, "model_fpath", None))
        have_tok = bool(getattr(args, "tokenizer_fpath", None))

        if not have_model and not have_tok:
            raise ValueError("Provide at least one of 'model_fpath' or 'tokenizer_fpath'.")

        if have_model and not os.path.exists(args.model_fpath):
            raise FileNotFoundError(
                f"Training warm-start checkpoint not found at '{args.model_fpath}'."
            )
        if have_tok and not os.path.exists(args.tokenizer_fpath):
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found at '{args.tokenizer_fpath}'."
            )

        # Load order: model first (if provided), then tokenizer (if provided)
        if have_model:
            model_state_dict = torch.load(args.model_fpath, map_location="cpu")
            model.load_state_dict(model_state_dict["model"])  # strict loading
            del model_state_dict

        if have_tok:
            tok_state = torch.load(args.tokenizer_fpath, map_location="cpu")
            if not hasattr(model, "tokenizer"):
                raise AttributeError("Model does not expose a 'tokenizer' attribute for loading.")
            model.tokenizer.load_state_dict(tok_state["model"])
            del tok_state
    else:
        # Tokenizer-only training: do not use model_fpath; optional tokenizer_fpath only
        if getattr(args, "tokenizer_fpath", None):
            if not os.path.exists(args.tokenizer_fpath):
                raise FileNotFoundError(
                    f"Tokenizer checkpoint not found at '{args.tokenizer_fpath}'."
                )
            tok_state = torch.load(args.tokenizer_fpath, map_location="cpu")
            model.load_state_dict(tok_state["model"])  # Tokenizer model
            del tok_state

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        cpu=False,
        log_with="wandb",
        kwargs_handlers=[kwargs],
        mixed_precision="bf16",
    )

    transforms = TransformsGenerator.get_final_transforms(model.image_size, None)

    def get_data_handlers(n_workers=8, n_envs=0, n_total_samples=10000):
        train_ds_mev = MultiEnvironmentDataset(
            dataset_folder,
            seq_length_input=num_frames - 1,
            seq_step=1,
            split_type="session",
            split="train",
            transform=transforms["train"],
            format=DatasetOutputFormat.IVG,
            enable_cache=args.train.enable_cache,
            cache_dpath=args.train.cache_dpath,
            n_workers=n_workers,
            n_envs=n_envs,
        )

        n_samples_valid = math.ceil(n_total_samples / train_ds_mev.n_datasets)

        valid_ds_mev = MultiEnvironmentDataset(
            dataset_folder,
            seq_length_input=num_frames - 1,
            seq_step=20,
            split_type="session",
            split="validation",
            transform=transforms["train"],
            format=DatasetOutputFormat.IVG,
            enable_cache=args.train.enable_cache,
            cache_dpath=args.train.cache_dpath,
            n_workers=n_workers,
            n_envs=n_envs,
            n_samples=n_samples_valid,
            source_dataset=train_ds_mev,
        )
        valid_ds_mev = Subset(valid_ds_mev, range(n_total_samples))

        return train_ds_mev, valid_ds_mev

    # check if it is the main thread
    if accelerator.is_main_process:
        train_ds, valid_ds = get_data_handlers(
            n_workers=n_workers, n_total_samples=n_total_samples, n_envs=n_envs
        )

    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        train_ds, valid_ds = get_data_handlers(
            n_workers=n_workers, n_total_samples=n_total_samples, n_envs=n_envs
        )

    save_dpath = f"{args.train.save_root_dpath}/{args.model}/{args.train.wandb_name}"

    # Ensure checkpoint directory exists
    os.makedirs(save_dpath, exist_ok=True)

    # Export full resolved Hydra config to YAML in the checkpoint directory
    try:
        OmegaConf.save(args, os.path.join(save_dpath, "config.yaml"), resolve=True)
    except Exception as e:
        print(f"Warning: failed to save config.yaml to '{save_dpath}': {e}")

    # For GenieRedux training (not tokenizer), create a symlink to the tokenizer weights
    if getattr(args, "model", None) != "tokenizer":
        link_path = os.path.join(save_dpath, "tokenizer.pt")
        try:
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.remove(link_path)
            tok_fpath = getattr(args, "tokenizer_fpath", None)
            if tok_fpath and os.path.exists(tok_fpath):
                os.symlink(os.path.abspath(tok_fpath), link_path)
        except Exception as e:
            print(f"Warning: failed to create tokenizer symlink at '{link_path}': {e}")

    trainer = Trainer(
        model=model,
        batch_size=args.train.batch_size,
        dataset=(train_ds, valid_ds),
        accelerator=accelerator,
        num_frames=args.train.num_frames,
        sample_num_frames=args.train.sample_num_frames,
        wandb_mode=args.train.wandb_mode,
        wandb_project=args.train.wandb_project,
        wandb_name=args.train.wandb_name,
        grad_accum_every=args.train.grad_accum,
        num_train_steps=args.train.num_train_steps,
        lr=args.optimizer.learning_rate,
        wd=args.optimizer.weight_decay,
        linear_warmup_start_factor=args.optimizer.linear_warmup_start_factor,
        linear_warmup_total_iters=args.optimizer.linear_warmup_total_iters,
        cosine_annealing_T_max=args.optimizer.cosine_annealing_t_max,
        cosine_annealing_eta_min=args.optimizer.cosine_annealing_min_lr,
        max_grad_norm=args.optimizer.max_grad_norm,
        save_dpath=save_dpath,
        save_model_every=args.train.save_model_every,
        wandb_dpath=args.train.wandb_dpath,
        max_valid_size=args.train.max_valid_size,
        validate_every=args.train.validate_every,
        train_config=args,
    )

    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        debug("config: ", args)
        print("GenieRedux training is starting...\n")
        print("Dataset : ", dataset_folder)
        count_parameters(model)

    if args.train.resume_ckpt != "no":
        trainer.load(args.train.resume_ckpt)
        print("Model loaded from file:", args.train.resume_ckpt)
        print("Starting training from step:", trainer.step)

    model.train()
    trainer.train()
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
