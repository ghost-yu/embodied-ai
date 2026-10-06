import os
from pathlib import Path
import hydra
from omegaconf import DictConfig

from tqdm import tqdm

from models import construct_model
from training.evaluation import Evaluator

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from einops import rearrange
import os
import torch
from data.data import (
    DatasetOutputFormat,
    TransformsGenerator,
    MultiEnvironmentDataset,
    video_tensor_to_gif,
    video_tensor_to_pil_images,
)
from torch.utils.data import DataLoader

from PIL import Image

from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs

import logging

logging.basicConfig(level=logging.INFO)
from tools.logger import getLogger

log = getLogger(__name__)


def get_inference_method(model, args, is_distributed=False):
    if args.model == "tokenizer":
        return model
    if "genie" in args.model:
        if args.eval.inference_method == "autoregressive":
            if is_distributed:
                sample_method = model.generate_interactive_video
            else:
                sample_method = model.module.generate_interactive_video
        elif args.eval.inference_method == "one_go":
            if is_distributed:
                sample_method = model.sample
            else:
                sample_method = model.module.sample

        return sample_method


def convert_index_to_one_hot(index, num_classes):
    one_hot = torch.zeros((*index.shape, num_classes), device=index.device)
    one_hot.scatter_(-1, index.unsqueeze(-1), 1)
    return one_hot


def generate_random_different_action_indices(actions_indices, device, num_actions=7):
    shape = actions_indices.shape
    random_actions = torch.randint(0, num_actions, shape, device=device)

    while torch.any(random_actions == actions_indices):
        random_actions = torch.where(
            random_actions == actions_indices,
            torch.randint(0, num_actions, shape, device=device),
            random_actions,
        )

    return random_actions


def evaluate(
    model,
    evaluator,
    test_loader,
    device,
    args,
    is_main_process=True,
    is_distributed=False,
):
    inference_method = get_inference_method(model, args, is_distributed)
    with torch.no_grad():
        psnr_scores = []
        ssim_scores = []
        delta_psnr_scores = []

        for i, videos in enumerate(
            tqdm(
                test_loader,
                total=len(test_loader),
                desc="Evaluating",
                disable=not is_main_process,
            )
        ):

            actions = videos["actions"]
            videos = videos["input_frames"]

            sample_num_frames = args.eval.sample_num_frames
            delta_psnr_horizon = args.eval.delta_psnr_horizon
            num_first_frames = args.eval.num_first_frames
            dream_length = args.eval.dream_length
            num_actions = args.eval.num_actions

            actions = actions.to(device)[:, : num_first_frames + sample_num_frames - 1]
            actions = actions.argmax(dim=-1)

            if args.eval.action_to_take != -1:
                actions = actions * 0 + args.eval.action_to_take

            videos = videos.to(device)[
                :,
                : num_first_frames + sample_num_frames,
            ]

            videos = rearrange(videos, "b f c h w -> b c f h w")
            first_frames = videos[:, :, :num_first_frames]

            recons = inference_method(
                videos=videos,
                prime_frames=first_frames,
                actions=actions,
                num_frames=sample_num_frames,
                inference_steps=args.eval.inference_steps,
                dream_length=dream_length,
                return_recons_only=True,
            )

            recons = torch.clamp(recons, min=0, max=1)
            videos = videos[:, :, num_first_frames:]
            recons_random = None
            delta_psnr = None
            if args.eval.eval_control:
                random_actions = generate_random_different_action_indices(
                    actions[:, : num_first_frames + delta_psnr_horizon - 1],
                    device,
                    num_actions=num_actions,
                )

                new_actions = actions[
                    :, : num_first_frames + delta_psnr_horizon - 1
                ].clone()
                new_actions[:, -1] = random_actions[:, -1]

                recons_random = inference_method(
                    videos=videos,
                    prime_frames=first_frames,
                    actions=new_actions,
                    num_frames=delta_psnr_horizon,
                    dream_length=dream_length,
                    return_recons_only=True,
                )
                recons_random = torch.clamp(recons_random, min=0, max=1)
                delta_psnr = evaluator.delta_psnr(
                    videos[:, :, delta_psnr_horizon - 1 : delta_psnr_horizon],
                    recons[:, :, delta_psnr_horizon - 1 : delta_psnr_horizon],
                    recons_random[:, :, -1:],
                )
                delta_psnr_scores.append(delta_psnr)

            evaluator.fid_update_batch(videos, recons)

            psnr = evaluator.psnr(videos, recons)
            ssim = evaluator.ssim(
                rearrange(videos, "b c f h w -> (b f) c h w"),
                rearrange(recons, "b c f h w -> (b f) c h w"),
            )
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

            log.i(f"Current scores: {psnr} PSNR; {ssim} SSIM, {delta_psnr} Delta PSNR")

            if i < 10 and is_main_process:
                sampled_videos_path = Path(args.eval.save_root_dpath) / f"{args.eval.dataset_name}/{args.eval.model_name}/samples_{i}"
                (sampled_videos_path).mkdir(parents=True, exist_ok=True)
                for j, recons_frames in enumerate(recons.unbind(dim=0)):
                    if j >= 10:
                        break
                    recons_frames = torch.cat([first_frames[j], recons_frames], dim=1)
                    orig_frames = torch.cat([first_frames[j], videos[j]], dim=1)
                    combined_frames = torch.cat([orig_frames, recons_frames], dim=2)

                    recon_frames = video_tensor_to_pil_images(
                        recons_frames.cpu(), only_first_image=False
                    )
                    orig_frames = video_tensor_to_pil_images(
                        orig_frames.cpu(), only_first_image=False
                    )

                    combined_height = orig_frames.height + recon_frames.height
                    combined_image = Image.new(
                        "RGB", (orig_frames.width, combined_height)
                    )

                    # Paste the images into the combined image
                    combined_image.paste(orig_frames, (0, 0))
                    combined_image.paste(recon_frames, (0, orig_frames.height))

                    video_tensor_to_gif(
                        combined_frames.cpu(),
                        str(sampled_videos_path / f"{j}.gif"),
                    )
                    combined_image.save(sampled_videos_path / f"{j}.png")

            if is_main_process:
                log.i(f"Batch {i} Evaluation done!")

    psnr_score = torch.mean(torch.tensor(psnr_scores, device=device))
    ssim_score = torch.mean(torch.tensor(ssim_scores, device=device))
    delta_psnr_score = torch.mean(torch.tensor(delta_psnr_scores, device=device))

    psnr_score = psnr_score.mean().item()
    ssim_score = ssim_score.mean().item()
    delta_psnr_score = delta_psnr_score.mean().item()
    fid_score = evaluator.fid()

    log.i(
        f"Device: {device},Average FID: {fid_score}, Average PSNR: {psnr_score}, Average SSIM: {ssim_score}, Average Delta PSNR: {delta_psnr_score}"
    )


@torch.no_grad()
def run(args):
    dataset_folder = f"{args.eval.dataset_root_dpath}/{args.eval.dataset_name}"

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[kwargs])

    device = accelerator.device
    evaluator = Evaluator(device)

    model = construct_model(args)

    # Harmonized load behavior with training
    if getattr(args, "model", None) != "tokenizer":
        have_model = bool(getattr(args.eval, "model_fpath", None))
        have_tok = bool(getattr(args, "tokenizer_fpath", None))

        if not have_model and not have_tok:
            raise ValueError("Provide at least one of eval.model_fpath or tokenizer_fpath for evaluation.")

        if have_model and not os.path.exists(args.eval.model_fpath):
            raise FileNotFoundError(
                f"Evaluation model checkpoint not found at '{args.eval.model_fpath}'."
            )
        if have_tok and not os.path.exists(args.tokenizer_fpath):
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found at '{args.tokenizer_fpath}'."
            )

        if have_model:
            model_state_dict = torch.load(args.eval.model_fpath, map_location="cpu")
            model.load_state_dict(model_state_dict["model"])  # strict
            del model_state_dict

        if have_tok:
            tok_state = torch.load(args.tokenizer_fpath, map_location="cpu")
            if not hasattr(model, "tokenizer"):
                raise AttributeError("Model does not expose a 'tokenizer' attribute for loading.")
            model.tokenizer.load_state_dict(tok_state["model"]) 
            del tok_state
    else:
        # Tokenizer-only eval: do not use eval.model_fpath; optional tokenizer_fpath only
        if getattr(args, "tokenizer_fpath", None):
            if not os.path.exists(args.tokenizer_fpath):
                raise FileNotFoundError(
                    f"Tokenizer checkpoint not found at '{args.tokenizer_fpath}'."
                )
            tok_state = torch.load(args.tokenizer_fpath, map_location="cpu")
            model.load_state_dict(tok_state["model"])  # Tokenizer model
            del tok_state

    transforms = TransformsGenerator.get_final_transforms(model.image_size, None)
    test_data_set = MultiEnvironmentDataset(
        dataset_folder,
        seq_length_input=args.eval.num_frames - 1,
        seq_step=args.eval.seq_step,
        split_type="instance",
        split="all",
        transform=transforms["test"],
        format=DatasetOutputFormat.IVG,
        enable_cache=bool(getattr(args.eval, "enable_cache", False)),
        n_workers=args.eval.n_data_workers,
        n_envs=args.eval.n_envs,
        cache_dpath=f"cache/evaluation/{args.eval.dataset_name}",
    )

    test_loader = DataLoader(
        test_data_set, batch_size=args.eval.batch_size, shuffle=False, num_workers=6
    )

    model, test_loader, evaluator = accelerator.prepare(model, test_loader, evaluator)

    is_main_process = accelerator.is_main_process
    is_distributed = accelerator.distributed_type == DistributedType.NO

    evaluate(
        model, evaluator, test_loader, device, args, is_main_process, is_distributed
    )


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
