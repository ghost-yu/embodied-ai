from collections import defaultdict
from functools import partial
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Dict, Optional, Tuple

from data_generation.generator.connector_retro_act import GameData, make_retro
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from tqdm import tqdm

from models.dynamics import Dynamics, MaskGIT
from models.genie_redux import GenieReduxGuided
from models.tokenizer import Tokenizer
from data.data import  TransformsGenerator

from .agent import AutoExploreAgent
from .collector import Collector
from .envs import SingleProcessEnv, MultiProcessEnv
from .models.actor_critic import ActorCritic
from .utils import configure_optimizer, EpisodeDirManager, set_seed
from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from .utils import FileStructure
from tools.model_management import CheckpointDirManager
import os

from tools.logger import getLogger
log = getLogger(__name__)

def get_game_list(root_dpath):
    game_data = GameData(annotation_fpath=os.path.join(root_dpath,"annotations.csv"), control_annotation_fpath=os.path.join(root_dpath, "controls.csv"), enable_sort=True)
    game_data.clean()
    selected_games = game_data.query(genre=["pl"], motion=None, view=None, game=None, platform=None)
    if len(selected_games) == 0:
        raise ValueError(f"No games found")
    return selected_games

def create_genie(model_fpath:Path, vq_loss_weight, recons_loss_weight):
    model_dpath = model_fpath.parent
    # Prefer YAML config exported by GenieRedux training; fallback to JSON if needed
    cfg_fpath = model_dpath / "config.yaml"
    if not cfg_fpath.exists():
        cfg_fpath = model_dpath / "config.yml"
    if not cfg_fpath.exists():
        cfg_fpath = model_dpath / "config.json"

    if not cfg_fpath.exists():
        raise FileNotFoundError(
            f"No config file found under '{model_dpath}'. Expected config.yaml|yml|json"
        )

    # Load Hydra config (YAML/JSON) and convert to a plain dict
    cfg_hydra = OmegaConf.load(cfg_fpath)
    config = OmegaConf.to_container(cfg_hydra, resolve=True)

    tokenizer_path = model_dpath / "tokenizer.pt"

    # Read from nested sections: tokenizer and dynamics
    t_cfg = config["tokenizer"]
    d_cfg = config["dynamics"]

    # Tokenizer-related
    t_dim = t_cfg["dim"]
    codebook_size = t_cfg["codebook_size"]
    t_image_size = t_cfg["image_size"]
    t_patch_size = t_cfg["patch_size"]
    t_temporal_patch_size = t_cfg["temporal_patch_size"]
    tokenizer_num_blocks = t_cfg["num_blocks"]
    t_dim_head = t_cfg["dim_head"]
    t_heads = t_cfg["heads"]
    t_ff_mult = t_cfg["ff_mult"]

    # Dynamics-related
    d_dim = d_cfg["dim"]
    action_dim = d_cfg["action_dim"]
    d_image_size = d_cfg["image_size"]
    d_patch_size = d_cfg["patch_size"]
    dynamics_num_blocks = d_cfg["num_blocks"]
    d_dim_head = d_cfg["dim_head"]
    d_heads = d_cfg["heads"]

    max_seq_len = d_cfg["max_seq_len"]
    sample_temperature = d_cfg["sample_temperature"]
    use_action_embeddings = d_cfg.get("use_action_embeddings", False)
    is_guided = d_cfg.get("is_guided", True)
    effective_action_dim = d_dim if use_action_embeddings else action_dim

    tokenizer = Tokenizer(
        dim=t_dim,  # embedding size
        codebook_size=codebook_size,  # codebook size
        image_size=t_image_size,  # H,W
        patch_size=t_patch_size,  # spatial patch size
        wandb_mode="enabled",
        temporal_patch_size=t_temporal_patch_size,  # temporal patch size
        num_blocks=tokenizer_num_blocks,  # nb of blocks in st transformer
        dim_head=t_dim_head,  # hidden size in transfo
        heads=t_heads,  # nb of heads for multi head transfo
        ff_mult=t_ff_mult,  # 32 * 64 = 2048 MLP size in transfo out
        vq_loss_w=vq_loss_weight,  # vq loss weight
        recon_loss_w=recons_loss_weight,  # reconstruction loss weight
    )

    # load the state dict of the tokenizer
    tokenizer_state_dict = torch.load(tokenizer_path, map_location=torch.device('cpu'))
    tokenizer.load_state_dict(tokenizer_state_dict["model"])

    maskgit = MaskGIT(
        dim=d_dim,
        is_guided=is_guided,
        action_dim=effective_action_dim,
        num_tokens=codebook_size,
        heads=d_heads,
        dim_head=d_dim_head,
        num_blocks=dynamics_num_blocks,
        max_seq_len=max_seq_len,
        image_size=d_image_size,
        patch_size=d_patch_size,
        use_token=d_cfg.get("use_token", True),
    )
    
    dynamics = Dynamics(
        maskgit=maskgit,
        inference_steps=50,
        mask_schedule="exp",
        sample_temperature=sample_temperature,
        use_distance_weighted_loss=False,
        use_focal_loss=False,
    )

    genie_guided: GenieReduxGuided = GenieReduxGuided(
        tokenizer=tokenizer,
        dynamics=dynamics,
        use_action_embeddings=use_action_embeddings
    )

    # load the state dict of the genie guided
    genie_guided_state_dict = torch.load(model_fpath, map_location="cpu")
    genie_guided.load_state_dict(genie_guided_state_dict["model"], strict=False)

    return genie_guided

import numpy as np
class MultiEnvWrapper:
    def __init__(self, fn_env_create, games):
        random.seed()
        np.random.seed()

        game = random.sample(games, 1)[0]
        log.i("Initializing:", game)
        self.env = fn_env_create(game=game)
        self.games = games
        self.fn_env_create = fn_env_create

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        self.env.close()
        game = random.sample(self.games, 1)[0]
        self.env = self.fn_env_create(game=game)
        return self.env.reset(**kwargs)

    def step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)


class Trainer:
    def __init__(self, cfg: DictConfig, fabric: Fabric, fs:FileStructure) -> None:

        wandb_logger = WandbLogger(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            **cfg.wandb
        )
        fabric.loggers.append(wandb_logger)

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(fabric.device) #cfg.common.device
        # Accumulate evaluation returns across epochs (for optional end-of-run summary)
        self.eval_returns: list[float] = []

        self.fabric = fabric
        device = fabric.device
        rank = fabric.global_rank
        world_size = fabric.world_size

        self.ckpt_dir = Path(fs.checkpoints_dpath)
        self.media_dir = Path(fs.media_dpath)
        #get the real absolute path of checkpoints and media
        self.ckpt_dir = self.ckpt_dir.resolve()
        self.media_dir = self.media_dir.resolve()
        #replace "home" with "scrath" in the path

        self.episode_dir = Path(fs.episodes_dpath)
        self.reconstructions_dir = Path(fs.reconstructions_dpath)

        if not cfg.common.resume:
            config_dir = Path(fs.config_dpath)
            config_dir = config_dir.resolve()
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=True, parents=True)
            log.d(f"Saving config to {config_path}")
            if fabric.is_global_zero:
                shutil.copy('.hydra/config.yaml', config_path)
                self.ckpt_dir.mkdir(exist_ok=False, parents=True)
                self.media_dir.mkdir(exist_ok=False, parents=True)
                self.episode_dir.mkdir(exist_ok=False, parents=False)
                self.reconstructions_dir.mkdir(exist_ok=False, parents=False)
            fabric.barrier()

        episode_manager_train = EpisodeDirManager(self.episode_dir / 'train', max_num_episodes=cfg.collection.train.num_episodes_to_save)
        episode_manager_test = EpisodeDirManager(self.episode_dir / 'test', max_num_episodes=cfg.collection.test.num_episodes_to_save)
        self.episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination', max_num_episodes=cfg.evaluation.actor_critic.num_episodes_to_save)

        # Track best evaluation reward and persist across runs if present
        self.best_eval_reward = float('-inf')
        try:
            best_val_path = self.ckpt_dir / 'best_reward_value.pt'
            if best_val_path.exists():
                self.best_eval_reward = float(torch.load(best_val_path))
        except Exception:
            pass

        # Define discrete action set once, used consistently by env and agent
        valid_action_combos = ["RIGHT", "LEFT", "UP", "DOWN", "ACTION_JUMP"]

        def create_env(cfg_env, num_envs, transform=None):
            games = cfg.collection.games
            max_episode_steps = cfg_env.max_episode_steps
            frame_skip = cfg_env.frame_skip

            def make_retro_multi(games):
                env_fn = partial(
                    make_retro,
                    render_mode="rgb_array",
                    valid_action_combos=valid_action_combos,
                    skip_frames=frame_skip,
                    max_episode_steps=max_episode_steps
                )
                env = MultiEnvWrapper(env_fn, games)
                return env
            
            fn_make_env = partial(make_retro_multi, games=games)

            env = MultiProcessEnv(fn_make_env, num_envs, should_wait_num_envs_ratio=0.5, transform=transform)
            return env

        model_fname = cfg.world_model.model_fname
        model_dname = cfg.world_model.model_dname
        checkpoint_root_dpath = cfg.world_model.root_dpath
        # Support numeric IDs via CheckpointDirManager for backward compatibility
        try:
            model_id_int = int(str(model_dname))
            cdm = CheckpointDirManager(checkpoint_root_dpath)
            model_dpath = cdm.get_dpath_by_id(model_id_int)
        except ValueError:
            model_dpath = Path(checkpoint_root_dpath) / str(model_dname)
        model_fpath = model_dpath / model_fname
        world_model = None
        world_model = create_genie(model_fpath=model_fpath, vq_loss_weight=1.0, recons_loss_weight=1.0)
        # Size the actor head to the environment's discrete action set
        actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=len(valid_action_combos))
        actor_critic = fabric.setup_module(actor_critic)
        actor_critic.mark_forward_method("reset")
        actor_critic.mark_forward_method("compute_loss")
        self.agent =  AutoExploreAgent(world_model, actor_critic).to(self.device)
        print(f'{sum(p.numel() for p in self.agent.actor_critic.parameters())} parameters in agent.actor_critic')

        self.optimizer_actor_critic = torch.optim.Adam(self.agent.actor_critic.parameters(), lr=cfg.training.learning_rate)
        self.optimizer_actor_critic = fabric.setup_optimizers(self.optimizer_actor_critic)

        
        transforms = TransformsGenerator.get_final_transforms(world_model.image_size, None)
        transform = transforms["train"]
        if self.cfg.training.should:
            train_env = []
            for id in cfg.env.train.id:
                cnfg = OmegaConf.create(cfg.env.train)
                cnfg.id = id
                train_env.append(create_env(cnfg, cfg.collection.train.num_envs, transform=transform))
            self.train_dataset = instantiate(cfg.datasets.train)
            self.train_collector = Collector(
                train_env,
                self.train_dataset,
                episode_manager_train,
                world_model=world_model,
                intrinsic_reward_type=cfg.collection.reward.type,
                entropy_top_fraction=cfg.collection.reward.entropy_top_fraction,
            )

        if self.cfg.evaluation.should and fabric.is_global_zero:
            test_env = []
            for id in cfg.env.test.id:
                cnfg = OmegaConf.create(cfg.env).test
                cnfg.id = id
                test_env.append(create_env(cnfg, cfg.collection.test.num_envs, transform=transform))
            self.test_dataset = instantiate(cfg.datasets.test)
            self.test_collector = Collector(
                test_env,
                self.test_dataset,
                episode_manager_test,
                world_model=world_model,
                intrinsic_reward_type=cfg.collection.reward.type,
                entropy_top_fraction=cfg.collection.reward.entropy_top_fraction,
            )

        assert self.cfg.training.should or self.cfg.evaluation.should


        if cfg.initialization.path_to_checkpoint is not None:
            self.agent.load(**cfg.initialization, device=self.device)

        if cfg.common.resume:
            self.load_checkpoint(cfg.common.resume_ckpt_id)

    def run(self) -> None:
        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

            if self.fabric.is_global_zero:
                log.i(f"Epoch {epoch} / {self.cfg.common.epochs}")
            start_time = time.time()
            to_log = []

            if self.cfg.training.should:
                if epoch <= self.cfg.collection.train.stop_after_epochs:
                    to_log += self.train_collector.collect(self.agent, epoch, use_print=self.fabric.is_global_zero, **self.cfg.collection.train.config)
                to_log += self.train_agent(epoch)

            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0) and self.fabric.is_global_zero:
                self.test_dataset.clear()
                eval_logs = self.test_collector.collect(self.agent, epoch, **self.cfg.collection.test.config, log_gifs=True)
                to_log += eval_logs
                # Save best model by evaluation reward
                maybe_return = self._extract_eval_return(eval_logs, key_prefix='test_dataset/')
                if maybe_return is not None:
                    # Optional per-epoch evaluation summary (enabled in eval_autoexplore)
                    if hasattr(self.cfg, 'evaluation') and getattr(self.cfg.evaluation, 'print_summary', False):
                        # keep track for end-of-run summary
                        self.eval_returns.append(maybe_return)
                        log.i("==== Evaluation Summary ====")
                        log.i(f"Average return: {maybe_return:.4f}")
                        log.i("===========================")
                if maybe_return is not None and maybe_return > self.best_eval_reward:
                    self.best_eval_reward = float(maybe_return)
                    try:
                        torch.save(self.agent.state_dict(), self.ckpt_dir / 'model_best_reward.pt')
                        torch.save(self.best_eval_reward, self.ckpt_dir / 'best_reward_value.pt')
                        log.i(f"New best eval return {self.best_eval_reward:.4f}. Saved model_best_reward.pt")
                    except Exception as e:
                        log.warning(f"Failed to save best model: {e}")

            self.fabric.barrier()
            if self.cfg.training.should and self.fabric.is_global_zero:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

            to_log.append({'duration': (time.time() - start_time) / 3600})
            for metrics in to_log:
                if self.fabric.is_global_zero:
                    self.fabric.log_dict({'epoch': epoch, **metrics}, step=self.cfg.training.actor_critic.steps_per_epoch  * (epoch - self.start_epoch + 1))

        # End-of-run evaluation summary across epochs (only when enabled)
        if (
            self.fabric.is_global_zero
            and len(self.eval_returns) > 0
            and hasattr(self.cfg, 'evaluation')
            and getattr(self.cfg.evaluation, 'print_summary', False)
        ):
            avg_all = sum(self.eval_returns) / len(self.eval_returns)
            log.i("==== Final Evaluation Summary ====")
            log.i(f"Epochs evaluated: {len(self.eval_returns)} | Average return across epochs: {avg_all:.4f}")
            log.i("==================================")
        self.finish()

    def _extract_eval_return(self, logs: list[dict], key_prefix: str = 'test_dataset/') -> Optional[float]:
        """Extract average return from evaluation logs.
        Looks for a key like 'test_dataset/return' and returns its value if found.
        """
        value = None
        for d in logs:
            for k, v in d.items():
                if isinstance(k, str) and k.startswith(key_prefix) and k.endswith('/return'):
                    try:
                        value = float(v)
                    except Exception:
                        continue
        return value

    def train_agent(self, epoch: int) -> None:
        self.agent.train()
        self.agent.zero_grad()

        metrics_tokenizer, metrics_world_model, metrics_actor_critic = {}, {}, {}

        
        cfg_actor_critic = self.cfg.training.actor_critic
        

        

        

        if epoch > cfg_actor_critic.start_after_epochs:
            self.agent.actor_critic.train()
            metrics_actor_critic = self.train_component(self.agent.actor_critic, self.optimizer_actor_critic, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False, world_model=self.agent.world_model, **cfg_actor_critic)
        self.agent.actor_critic.eval()
        metrics_actor_critic = {f'{str(self.agent.actor_critic.module)}/train/{k}': v for k, v in metrics_actor_critic.items()}
        return [{'epoch': epoch, **metrics_tokenizer, **metrics_world_model, **metrics_actor_critic}]

    def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, steps_per_epoch: int, batch_num_samples: int, grad_acc_steps: int, max_grad_norm: Optional[float], sequence_length: int, sample_from_start: bool, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)
        assert batch_num_samples % self.fabric.world_size == 0
        batch_num_samples = batch_num_samples // self.fabric.world_size
        component_name = str(component.module)

        #create a data loader
        

        for _ in tqdm(range(steps_per_epoch), desc=f"Training {component_name}", file=sys.stdout, disable=not self.fabric.is_global_zero):

            optimizer.zero_grad()
            for i in range(grad_acc_steps):
                batch = self.train_dataset.sample_batch(batch_num_samples, sequence_length, sample_from_start)
                batch = self._to_device(batch)

                with self.fabric.no_backward_sync(component, enabled = i!=(grad_acc_steps-1)):
                    losses = component.compute_loss(batch, **kwargs_loss) / grad_acc_steps
                    loss_total_step = losses.loss_total
                    self.fabric.backward(loss_total_step)
                    loss_total_epoch += loss_total_step.item() / steps_per_epoch

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{component_name}/train/{loss_name}"] += loss_value / steps_per_epoch

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)

            optimizer.step()
        log.i("Final loss:", loss_total_epoch)
        metrics = {f'{component_name}/train/total_loss': loss_total_epoch, **intermediate_losses}
        return metrics

    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        if epoch % 25 == 0:
            torch.save(self.agent.state_dict(), self.ckpt_dir / f'{epoch}.pt')
        torch.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')
        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir / 'epoch.pt')
            torch.save({
                "optimizer_actor_critic": self.optimizer_actor_critic.state_dict(),
            }, self.ckpt_dir / 'optimizer.pt')
            ckpt_dataset_dir = self.ckpt_dir / 'dataset'
            ckpt_dataset_dir.mkdir(exist_ok=True, parents=False)
            self.train_dataset.update_disk_checkpoint(ckpt_dataset_dir)
            if self.cfg.evaluation.should:
                torch.save(self.test_dataset.num_seen_episodes, self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        # Save directly into checkpoint directory
        self._save_checkpoint(epoch, save_agent_only)

    def load_checkpoint(self, ckpt_id = "last") -> None:
        assert self.ckpt_dir.is_dir()
        self.start_epoch = torch.load(self.ckpt_dir / 'epoch.pt') + 1
        self.agent.load(self.ckpt_dir / f'{ckpt_id}.pt', device=self.device)
        ckpt_opt = torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
        self.optimizer_actor_critic.load_state_dict(ckpt_opt['optimizer_actor_critic'])
        if self.cfg.training.should:
            self.train_dataset.load_disk_checkpoint(self.ckpt_dir / 'dataset')
        if self.cfg.evaluation.should and (self.ckpt_dir / 'num_seen_episodes_test_dataset.pt').exists():
            self.test_dataset.num_seen_episodes = torch.load(self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')
        print(f'Successfully loaded model, optimizer.')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        #print all data types in the batch
        return {k: batch[k].to(self.device) for k in batch}

    def finish(self) -> None:
        pass
