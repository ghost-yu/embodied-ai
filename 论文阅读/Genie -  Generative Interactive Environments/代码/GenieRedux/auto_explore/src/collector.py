import os
import random
import sys
from typing import List, Optional, Union

import cv2
from einops import rearrange
from auto_explore.src.models.intrinsic import SmallConvNet, VanillaVAE
import numpy as np
import torch
from tqdm import tqdm
from data.data import TransformsGenerator
import wandb
import torch.nn.functional as F

from models.genie_redux import GenieReduxGuided
from .agent import AutoExploreAgent
from .dataset import EpisodesDataset
from .envs import SingleProcessEnv, MultiProcessEnv
from .episode import Episode
from .utils import EpisodeDirManager, RandomHeuristic
from lovely_numpy import lo
from tools.logger import getLogger
from PIL import Image
log = getLogger(__name__)


def process_obs_np(obs, transform):
    new_obs = []
    for ob in obs:
        ob = transform(ob)
        new_obs.append(ob)
    return new_obs

def process_obs(obs, dtype, transform, device):
    #obs: b h w c
    new_obs = process_obs_np(obs, transform)
    obs = torch.tensor(np.array(new_obs))
    obs = obs.to(device)
    obs = obs.to(dtype)
    return obs

class GifSaver:
    def __init__(self, root_dpath) -> None:
        self.frames = []
        self.root_dpath = os.path.abspath(root_dpath)
        os.makedirs(self.root_dpath, exist_ok=True)

    def add_frame(self, image: np.ndarray) -> None:
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image*255).astype(np.uint8))
        self.frames.append(image)

    def save_gif(self, save_path, duration: int = 100) -> str:
        save_path = f'{self.root_dpath}/{save_path}'
        log.d(f'Saving gif to {save_path}')
        if self.frames:
            self.frames[0].save(
                save_path,
                save_all=True,
                append_images=self.frames[1:],
                duration=duration,
                loop=0
            )
        return save_path

class Collector:
    def __init__(self, env: List[Union[SingleProcessEnv, MultiProcessEnv]]|Union[SingleProcessEnv, MultiProcessEnv], dataset: EpisodesDataset, episode_dir_manager: EpisodeDirManager, world_model:GenieReduxGuided=None, intrinsic_reward_type: str = 'entropy', entropy_top_fraction: float = 0.25) -> None:
        self.envs = env if isinstance(env, list) else [env]
        self.dataset = dataset
        self.episode_dir_manager = episode_dir_manager
        self.world_model = world_model
        self.env_step = 0
        self.intrinsic_reward_type = intrinsic_reward_type  # choose one of: 'entropy', 'cnn-mse', 'vae-mse'
        self.entropy_top_fraction = entropy_top_fraction

        if self.intrinsic_reward_type == 'cnn-mse':
            self.cnn = SmallConvNet(feat_dim=512, last_nl=None, layernormalize=False).to("cuda")
            # Use inference-only mode for CNN featurizer
            self.cnn.eval()
            for p in self.cnn.parameters():
                p.requires_grad = False
        elif self.intrinsic_reward_type == 'vae-mse':
            self.vae = VanillaVAE(in_channels=3, latent_dim=512).to("cuda")
            self.optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr=0.0001)

        self.env_id = random.randint(0, len(self.envs) - 1)
        self.env = self.envs[self.env_id]
        self.obs = self.env.reset()
        self.last_obs = []
        self.last_actions = []

        self.episode_ids = [None] * self.env.num_envs
        self.heuristic = RandomHeuristic(self.env.num_actions)

    def collect(self, agent: AutoExploreAgent, epoch: int, epsilon: float, should_sample: bool, temperature: float, burn_in: int, num_steps: Optional[int] = None, num_episodes: Optional[int] = None, min_epsilon: float = 0.01, n_preds: int = 3, use_print: bool = True, log_gifs: bool = False) -> List[dict]:
        assert 0 <= epsilon <= 1
        end_epoch = 300
        epoch_rate = max(0, end_epoch - epoch) / end_epoch
        epsilon = epsilon * epoch_rate + min_epsilon * (1 - epoch_rate)

        assert (num_steps is None) != (num_episodes is None)
        should_stop = lambda steps, episodes: steps >= num_steps if num_steps is not None else episodes >= num_episodes
        log.i(f"Collect targets -> num_steps={num_steps}, num_episodes={num_episodes}")

        to_log = []
        steps, episodes = 0, 0
        returns = []
        observations, actions, rewards, dones = [], [], [], []

        # Avoid training-time behavior and graph building during collection
        was_training = agent.actor_critic.training
        agent.actor_critic.eval()
        agent.actor_critic.reset(n=self.env.num_envs, burnin_observations=None, mask_padding=None) #TODO: butnin_obs_rec, mask_padding
        pbar = tqdm(total=num_steps if num_steps is not None else num_episodes, desc=f'Experience collection ({self.dataset.name})', file=sys.stdout, disable=not use_print)

        n_input_frames = 2
        n_preds = n_preds if self.intrinsic_reward_type == 'entropy' else 1
        device = agent.device

        os.makedirs('./outputs/gifs/', exist_ok=True)
        gif_saver = GifSaver('./outputs/gifs/')
        gif_saver.add_frame(self.obs[0])

        while not should_stop(steps, episodes):
            observations.append(self.obs)
            self.last_obs.append(self.obs)
            self.last_obs = self.last_obs[-(n_input_frames + n_preds):]
            obs = rearrange(torch.FloatTensor(self.obs), 'n h w c -> n c h w').to(agent.device)
            
            act = agent.act(obs, should_sample=should_sample, temperature=temperature).cpu().numpy()

            if random.random() < epsilon:
                act = self.heuristic.act(obs).cpu().numpy()
            
            #create one hot encoding of act
            act_one_hot = F.one_hot(torch.tensor(act), num_classes=self.env.num_actions).float().numpy()
            self.obs, reward_env, done, _ = self.env.step(act_one_hot) #B x H x W x C
            self.env_step += 1
            actions.append(act)
            self.last_actions.append(act)
            self.last_actions = self.last_actions[-(n_input_frames + n_preds)+1:]
            dones.append(done)
            # Append current observation frame to GIF
            try:
                gif_saver.add_frame(self.obs[0])
            except Exception:
                pass

            #calculate reward from world model uncertainty
            if self.world_model is not None:
                if len(self.last_obs) >= n_input_frames + n_preds:
                    with torch.no_grad():
                        wm_observations = np.array(self.last_obs[-(n_input_frames + n_preds):-n_preds])
                        t = wm_observations.shape[0]
                        

                        wm_observations = torch.tensor(wm_observations, dtype=torch.float32).to(agent.device)
                        wm_observations = rearrange(wm_observations, 't b h w c -> b c t h w', t=t)

                        wm_actions = np.array(self.last_actions[-(n_input_frames + n_preds)+1:]) #T x B
                        wm_actions = rearrange(wm_actions, 't b -> b t')
                        wm_actions = torch.from_numpy(wm_actions).to(agent.device)

                        if self.intrinsic_reward_type == 'entropy':
                            logits = self.world_model.sample(
                                prime_frames=wm_observations,
                                actions=wm_actions,
                                num_frames=n_preds,
                                inference_steps=1,
                                mask_schedule="exp",
                                sample_temperature=1.0,
                                return_logits=True,)
                        elif self.intrinsic_reward_type == 'cnn-mse':
                            output = self.world_model.sample(
                                prime_frames=wm_observations,
                                actions=wm_actions,
                                num_frames=n_preds,
                                inference_steps=1,
                                mask_schedule="exp",
                                sample_temperature=1.0,
                                return_logits=False,)
                            
                            wm_gt = np.array(self.last_obs[-n_preds:])
                            t = wm_gt.shape[0]
                            wm_gt = torch.tensor(wm_gt, dtype=torch.float32).to(agent.device)
                            wm_gt = rearrange(wm_gt, 't b h w c -> b c t h w', t=t)

                            with torch.no_grad():
                                feats_orig = self.cnn(wm_gt)
                                feats_pred = self.cnn(output[:,:,-n_preds:])
                        elif self.intrinsic_reward_type == 'vae-mse':
                            output = self.world_model.sample(
                                prime_frames=wm_observations,
                                actions=wm_actions,
                                num_frames=n_preds,
                                inference_steps=1,
                                mask_schedule="exp",
                                sample_temperature=1.0,
                                return_logits=False,)
                    
                    if self.intrinsic_reward_type == 'vae-mse':
                        
                        wm_gt = np.array(self.last_obs[-n_preds:])
                        t = wm_gt.shape[0]
                        t_output = output.shape[2]
                        b = output.shape[0]
                        wm_gt = torch.tensor(wm_gt, dtype=torch.float32).to(agent.device)
                        wm_gt = rearrange(wm_gt, 't b h w c -> b c t h w', t=t)
                        wm_gt = rearrange(wm_gt, 'b c t h w -> (b t) c h w')
                        output = rearrange(output, 'b c t h w -> (b t) c h w')



                        self.optimizer_vae.zero_grad()
                        self.vae.train()
                        vae_out = self.vae(torch.cat([wm_gt, output], dim=0))
                        recon, input, mu, log_var = vae_out
                        loss = self.vae.loss_function(recon, input, mu, log_var,  M_N = 0.005)
                        loss = loss["loss"]
                        loss.backward()
                        self.optimizer_vae.step()

                        mu_gt = mu[:t*b].detach()
                        mu_pred = mu[-t_output*b:].detach()

                        mu_gt = rearrange(mu_gt, '(b t) c -> b t c', t=t)
                        mu_pred = rearrange(mu_pred, '(b t) c -> b t c', t=t_output)

                        reward = ((mu_gt - mu_pred) ** 2).sum(axis=(1, 2)).cpu().numpy()
                    
                    # We only include input frames in the GIF (one frame per step).
                    # To include world-model context strips, re-enable the block below.

                    if self.intrinsic_reward_type == 'entropy':
                        with torch.no_grad():
                            logits = rearrange(logits, 'b (t d) c -> b t d c', t=n_preds)[:,-1]
                            b = logits.shape[0]
                            logits_categorical = rearrange(logits, 'b d c -> (b d) c')
                            categorical = torch.distributions.Categorical(logits=logits_categorical)
                            entropy = categorical.entropy()
                            entropy = (2.0*entropy)/np.log(logits_categorical.shape[-1])
                            entropy = rearrange(entropy, '(b d) -> b d', b=b)
                            #sort on the second dimension based on entropy value
                            entropy, _ = entropy.sort(dim=1, descending=True)
                            #take the mean of the top X% of the entropy values
                            top_k = max(1, int(entropy.shape[1] * float(self.entropy_top_fraction)))
                            entropy = entropy[:, :top_k].mean(dim=-1)

                            entropy = entropy.cpu().numpy()
                            entropy = np.round(entropy, 2)
                            reward = np.array(entropy)
                    elif self.intrinsic_reward_type == 'cnn-mse':
                        reward = ((feats_orig - feats_pred) ** 2).sum(dim=[1, 2]).cpu().numpy()

                else:
                    reward = np.zeros_like(reward_env) #+ np.array(reward)
            # Update progress bar with action and reward snapshot
            try:
                a0 = int(act[0]) if hasattr(act, '__len__') and len(act) > 0 else int(act)
                r_mean = float(np.mean(reward)) if reward is not None else 0.0
                pbar.set_description(f'Experience collection ({self.dataset.name}) | a0={a0} r={r_mean:.3f}')
            except Exception:
                pass

            rewards.append(reward)

            new_steps = len(self.env.mask_new_dones)
            steps += new_steps
            pbar.update(new_steps if num_steps is not None else 0)

            # Warning: with EpisodicLifeEnv + MultiProcessEnv, reset is ignored if not a real done.
            # Thus, segments of experience following a life loss and preceding a general done are discarded.
            # Not a problem with a SingleProcessEnv.

            if self.env.should_reset():
                self.last_obs = []
                self.last_actions = []
                log.i("Environment is resetting...")
                condition = np.zeros((1, len(self.envs)), dtype=int)
                condition[0, self.env_id] = 1
                gif_path = gif_saver.save_gif(f'video_epoch_{epoch}_step_{steps}.gif')
                gif_saver.frames.clear()
                if log_gifs:
                    try:
                        to_log.append({f'{self.dataset.name}/gif': wandb.Video(gif_path, fps=10, format='gif')})
                    except Exception:
                        pass
                self.add_experience_to_dataset(observations, actions, rewards, dones, condition)

                try:
                    done_sum = int(np.array(done).sum())
                except Exception:
                    done_sum = -1
                new_episodes = self.env.num_envs
                log.i(f"Episodes increment -> using num_envs={new_episodes} (done_sum_last_step={done_sum})")
                episodes += new_episodes
                pbar.update(new_episodes if num_episodes is not None else 0)

                for episode_id in self.episode_ids:
                    episode = self.dataset.get_episode(episode_id)
                    metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
                    metrics_episode['episode_num'] = episode_id
                    metrics_episode['action_histogram'] = wandb.Histogram(np_histogram=np.histogram(episode.actions.numpy(), bins=np.arange(0, self.env.num_actions + 1) - 0.5, density=True))
                    to_log.append({f'{self.dataset.name}/{k}': v for k, v in metrics_episode.items()})
                    returns.append(metrics_episode['episode_return'])

                env_id = random.randint(0, len(self.envs) - 1)
                self.env = self.envs[env_id]
                self.env_id = env_id
                self.obs = self.env.reset()
                self.env_step = 0
                self.episode_ids = [None] * self.env.num_envs
                agent.actor_critic.reset(n=self.env.num_envs)
                observations, actions, rewards, dones = [], [], [], []

        # Add incomplete episodes to dataset, and complete them later.
        if len(observations) > 0:
            condition = np.zeros((1, len(self.envs)), dtype=int)
            condition[0, self.env_id] = 1
            self.add_experience_to_dataset(observations, actions, rewards, dones, condition)

        agent.actor_critic.clear()
        log.i(f"Finished collection -> steps={steps}, episodes={episodes}")
        # Restore training mode if it was set
        if was_training:
            agent.actor_critic.train()

        metrics_collect = {
            '#episodes': len(self.dataset),
            '#steps': sum(map(len, self.dataset.episodes)),
        }
        log.i("Returns:", returns)
        if len(returns) > 0:
            metrics_collect['return'] = np.mean(returns)
        metrics_collect = {f'{self.dataset.name}/{k}': v for k, v in metrics_collect.items()}
        to_log.append(metrics_collect)

        return to_log

    def add_experience_to_dataset(self, observations: List[np.ndarray], actions: List[np.ndarray], rewards: List[np.ndarray], dones: List[np.ndarray], condition: np.ndarray) -> None:
        assert len(observations) == len(actions) == len(rewards) == len(dones)
        for i, (o, a, r, d) in enumerate(zip(*map(lambda arr: np.swapaxes(arr, 0, 1), [observations, actions, rewards, dones]))):  # Make everything (N, T, ...) instead of (T, N, ...)
            episode = Episode(
                observations=torch.ByteTensor(o).permute(0, 3, 1, 2).contiguous(),  # channel-first
                actions=torch.LongTensor(a),
                rewards=torch.FloatTensor(r),
                ends=torch.LongTensor(d),
                mask_padding=torch.ones(d.shape[0], dtype=torch.bool),
                condition=torch.IntTensor(condition),
            )
            if self.episode_ids[i] is None:
                self.episode_ids[i] = self.dataset.add_episode(episode)
            else:
                self.dataset.update_episode(self.episode_ids[i], episode)
