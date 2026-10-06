import argparse
import multiprocessing
import os
from typing import Tuple

from .connector_base import BaseConnector

# Be safe if context is already set by parent code
try:
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("fork")
except Exception:
    # Ignore if context already set or unsupported platform
    pass

import logging
import random

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd

# from stable_baselines3 import PPO
# from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
# from stable_baselines3.common.vec_env import (
#     SubprocVecEnv,
#     VecFrameStack,
#     VecTransposeImage,
# )
import retro
from pathlib import Path
import sys, types, importlib

# Lightweight helper to expose a namespace package without touching sys.path
repo_root = Path(__file__).resolve().parents[2]

def _expose_namespace(pkg_name: str, rel_path: str) -> None:
    """Expose `pkg_name` as a namespace whose __path__ points to repo_root/rel_path.

    Safe to call repeatedly; only sets up when missing. Avoids mutating sys.path.
    """
    if pkg_name in sys.modules and hasattr(sys.modules[pkg_name], "__path__"):
        return
    module = types.ModuleType(pkg_name)
    module.__path__ = [str(repo_root / rel_path)]
    sys.modules[pkg_name] = module

def ensure_auto_explore_imports() -> None:
    """Expose only the namespaces required by the AutoExplore connector.

    Keeps side effects minimal and scoped to when this connector is used.
    """
    _expose_namespace("auto_explore", "auto_explore")
    _expose_namespace("data", "data")                 # prefer repo-root data/
    _expose_namespace("data_generation", "data_generation")  # used by data.data
    _expose_namespace("tools", "tools")               # tools.logger
    _expose_namespace("models", "models")             # models.genie_redux

def import_string(path: str):
    """Import a module or object given a string.

    Examples:
    - "pkg.mod" -> returns the module object
    - "pkg.mod:ClassName" -> returns the attribute from the module
    - "pkg.mod.attr" -> returns the module (standard importlib semantics)
    """
    if ":" in path:
        mod_name, attr_name = path.split(":", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr_name)
    return importlib.import_module(path)


def ensure_deep_rl_zoo_imports() -> None:
    """Expose vendored deep_rl_zoo as a top-level package.

    The repo contains the package at data_generation/deep_rl_zoo/deep_rl_zoo.
    This shim registers a namespace so imports like `from deep_rl_zoo import greedy_actors`
    resolve correctly without modifying sys.path.
    """
    _expose_namespace("deep_rl_zoo", "data_generation/external/deep_rl_zoo/deep_rl_zoo")

# from gymnasium.wrappers.time_limit import TimeLimit

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(processName)s: %(message)s")


class GameData:
    def __init__(
        self,
        annotation_fpath: str = "annotations/RetroAct_v0.1.csv",
        control_annotation_fpath: str = "annotations/RetroAct_v0.1_control_GenieRedux-G-50_sublist.csv",
        enable_sort=False,
    ):
        self.annotation_fpath = annotation_fpath
        # read the csv into a pandas dataframe
        self.df = pd.read_csv(annotation_fpath, delimiter=",")
        # separate df["tags"] using splitting by space into three columns: view, motion and genre
        # if "new" is in tags remove from the data frame
        # sort by game
        if enable_sort:
            self.df = self.df.sort_values(by="game")
        self.df = self.df[~self.df["tags"].str.contains("new")]
        self.df[["view", "motion", "genre"]] = self.df["tags"].str.split(
            " ", expand=True
        )[[0, 1, 2]]
        self.df[["platform"]] = self.df["game"].str.split("-", expand=True)[[1]]

        if control_annotation_fpath is not None:
            self.enable_controls = True
            self.df["game_upper"] = self.df["game"]
            self.df["game"] = self.df["game"].str.lower()
            self.df_controls = pd.read_csv(control_annotation_fpath, delimiter=",")
            if enable_sort:
                self.df_controls = self.df_controls.sort_values(by="game")

            # convert the transition key from YES and NO values to 1 and 0
            self.df_controls["transition"] = self.df_controls["transition"].apply(
                lambda x: 1 if x == "YES" else 0,
            )
            # join the two dataframes on the game column but the game column in df is with some capitilized letters while in df_controls it is all lower case. After joining keep the capitalization
            # copy the game column in df to a new one called game_upper

            self.df = self.df.merge(
                self.df_controls,
                on="game",
                how="left",
                suffixes=("", "_controls"),
            )
            self.df["game"] = self.df["game_upper"]
            self.df = self.df.drop(columns=["game_upper"])
        else:
            self.enable_controls = False

    def filter(self, action_map):
        # clean actions
        if self.enable_controls:
            # remove all entries with ACTION_JUMP different than jump
            self.df = self.df[self.df["ACTION_JUMP"] == "jump"]
            # remove all entries with DOWN different than down crouch, climb down or  climb contained in the entry
            self.df = self.df[self.df["DOWN"].str.contains("none|crouch|climb")]
            # remove all entries with UP different than climb or none
            self.df = self.df[self.df["UP"].str.contains("climb|none")]
            # remove all entries with LEFT different than left
            self.df = self.df[self.df["LEFT"] == "left"]
            # remove all entries with RIGHT different than right
            self.df = self.df[self.df["RIGHT"] == "right"]
            # remove all entries with transition equal to 1
            # self.df = self.df[self.df["transition"] == 0]

    def query(self, view=None, motion=None, genre=None, game=None, platform=None):
        df = self.df
        if view is not None:
            df = df[df["view"].isin(view)]
        if motion is not None:
            df = df[df["motion"].isin(motion)]
        if genre is not None:
            df = df[df["genre"].isin(genre)]
        if game is not None:
            df = df[df["game"].isin(game)]
        if platform is not None:
            df = df[df["platform"].isin(platform)]
        return df["game"].tolist()


class OneHotEncoding(gym.Space):
    """

    Based on: https://stackoverflow.com/questions/54022606/openai-gym-how-to-create-one-hot-observation-space
    {0,...,1,...,0}

    Example usage:
    self.observation_space = OneHotEncoding(size=4)
    """

    def __init__(self, size=None):
        assert isinstance(size, int) and size > 0
        self.size = size
        self.n = size
        gym.Space.__init__(self, (self.size,), np.int64)

    def sample(self):
        one_hot_vector = np.array([0] * self.size)  # np.zeros(self.size)
        one_hot_vector[np.random.randint(self.size)] = 1
        return one_hot_vector

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            number_of_zeros = list(x).count(0)
            number_of_ones = list(x).count(1)
            return (number_of_zeros == (self.size - 1)) and (number_of_ones == 1)
        return False

    def __repr__(self):
        return "OneHotEncoding(%d)" % self.size

    def __eq__(self, other):
        return self.size == other.size


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use one hot discrete actions.
    based on https://gist.github.com/christopherhesse/8e5c63c3b0007f4c3e333c6a158872cf
    Args:
        combos: ordered list of lists of valid button combinations
    """

    REMAPPER = {
        "atari2600": {
            "A": "BUTTON",
            "B": "BUTTON",
            "ACTION_JUMP": "BUTTON",
            "ACTION_PRIMARY": "BUTTON",
            "ACTION_SECONDARY": "BUTTON",
        },
        "nes": {"ACTION_JUMP": "A", "ACTION_PRIMARY": "A", "ACTION_SECONDARY": "B"},
        "snes": {"ACTION_JUMP": "B", "ACTION_PRIMARY": "B", "ACTION_SECONDARY": "Y"},
        "genesis": {"ACTION_JUMP": "C", "ACTION_PRIMARY": "C", "ACTION_SECONDARY": "B"},
        "sms": {"ACTION_JUMP": "A", "ACTION_PRIMARY": "A", "ACTION_SECONDARY": "B"},
        "gameboy": {"ACTION_JUMP": "A", "ACTION_PRIMARY": "A", "ACTION_SECONDARY": "B"},
        "32x": {"ACTION_JUMP": "B", "ACTION_PRIMARY": "B", "ACTION_SECONDARY": "Y"},
    }

    def __init__(self, env, platform, combos=None):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        # if len(buttons) == 0:
        #     buttons = ['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']
        self._decode_discrete_action = []
        self.combos = combos
        self.noop_id = 0

        if self.combos is None:
            self.combos = buttons

        self.combos = [
            combo if isinstance(combo, list) else [combo] for combo in self.combos
        ]

        try:
            for combo in self.combos:
                arr = np.array([0] * env.action_space.n)
                for button in combo:
                    button = self.REMAPPER[platform.lower()].get(button, button)
                    if button.upper() != "NOOP":
                        arr[buttons.index(button)] = 1
                    # arr[button] = 1

                self._decode_discrete_action.append(arr)
        except:
            raise ValueError(f"Invalid action combination {buttons}")
        self.action_space = OneHotEncoding(len(self._decode_discrete_action))

    def action(self, act):
        # check if act is an array made out of zeros
        act = np.argmax(act)
        return self._decode_discrete_action[act].copy()


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob, want_render=True):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")
        self.want_render = want_render

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            elif self.supports_want_render:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=self.want_render,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew

            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(
    *,
    game,
    state=None,
    max_episode_steps=None,
    skip_frames=1,
    render_mode="human",
    use_discretization=True,
    valid_action_combos=None,
    **kwargs,
):
    if state is None:
        state = retro.State.DEFAULT
    if skip_frames < 1:
        raise ValueError(f"skip_frames must be at least 1, got {skip_frames}")

    env = retro.make(game, state, render_mode=render_mode, **kwargs)
    env = StochasticFrameSkip(env, n=skip_frames, stickprob=0, want_render=True)
    if max_episode_steps is not None:
        raise NotImplementedError(
            "TimeLimit wrapper is not implemented for retro environments. "
            "Please use a custom TimeLimit wrapper if needed.",
        )
        # env = TimeLimit(env, max_episode_steps=max_episode_steps)

    if use_discretization:
        platform = game.split("-")[1]
        env = Discretizer(env, platform, combos=valid_action_combos)
    return env


def pad_to_match_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int]):
    height, width = image.shape[:2]
    target_width, target_height = target_size

    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height

    # Determine padding
    if aspect_ratio > target_aspect_ratio:
        # Width is larger than target, pad top and bottom
        new_width = width
        new_height = int(width / target_aspect_ratio)
        top_pad = (new_height - height) // 2
        bottom_pad = new_height - height - top_pad
        pad_width = ((top_pad, bottom_pad), (0, 0), (0, 0))
    else:
        # Height is larger than target, pad left and right
        new_height = height
        new_width = int(height * target_aspect_ratio)
        left_pad = (new_width - width) // 2
        right_pad = new_width - width - left_pad
        pad_width = ((0, 0), (left_pad, right_pad), (0, 0))

    # Pad the image
    if len(image.shape) == 3:
        # Image has channels (e.g., RGB)
        padded_image = np.pad(image, pad_width, mode="constant", constant_values=0)
    else:
        # Grayscale image
        pad_width = pad_width[:2]
        padded_image = np.pad(image, pad_width, mode="constant", constant_values=0)

    return padded_image


# create a new Exception called TrackerError
class TrackerError(Exception):
    pass


def apply_tracker(tracker, frame, box_size):
    (H, W) = frame.shape[:2]
    (success, box) = tracker.update(frame)
    # check to see if the tracking was a success
    if not success:
        raise TrackerError("Failed to track the object")

    (x, y, w, h) = [int(v) for v in box]
    # cv2.rectangle(frame, (x, y), (x + w, y + h),
    #     (0, 255, 0), 2)

    # calculate the center of the bounding box
    centerX, centerY = x + w // 2, y + h // 2
    # define the size of the small box
    small_box_size = box_size
    half_small_box_size = small_box_size // 2
    # calculate the coordinates of the small box
    small_box_x1 = centerX - half_small_box_size
    small_box_y1 = centerY - half_small_box_size
    small_box_x2 = centerX + half_small_box_size
    small_box_y2 = centerY + half_small_box_size

    # create a black canvas for the small box
    small_box = np.zeros((small_box_size, small_box_size, 3), dtype="uint8")

    # calculate the coordinates for placing the frame on the canvas
    startX = max(0, -small_box_x1)
    startY = max(0, -small_box_y1)
    endX = small_box_size - max(0, small_box_x2 - W)
    endY = small_box_size - max(0, small_box_y2 - H)

    # calculate the coordinates for cropping the frame
    cropX1 = max(0, small_box_x1)
    cropY1 = max(0, small_box_y1)
    cropX2 = min(W, small_box_x2)
    cropY2 = min(H, small_box_y2)

    # place the cropped frame on the canvas
    small_box[startY:endY, startX:endX] = frame[cropY1:cropY2, cropX1:cropX2]

    # create a mask of the region that remains black in small box
    mask = np.zeros((small_box_size, small_box_size), dtype="uint8")
    mask[startY:endY, startX:endX] = 1

    return small_box, mask


class RetroActConnector(BaseConnector):
    def __init__(self, config=None):
        if config is None:
            config = {
                "name": "retro_act",
                "version": "0.1.0",
                "game": "Airstriker-Genesis",
                "n_skip_frames": 1,
                "n_skip_start_frames": 0,
                "image_size": None,
                "valid_action_combos": None,
                "use_tracker": False,
                "roi_annotation_fpath": None,
            }

        self.config = config

        self.name = config["name"]
        self.version = config["version"]
        self.game = config["game"]
        self.n_skip_frames = config["n_skip_frames"]
        self.n_skip_start_frames = config["n_skip_start_frames"]
        self.image_size = config["image_size"]
        self.valid_action_combos = config["valid_action_combos"]
        self.roi_annotation_fpath = config["roi_annotation_fpath"]
        self.use_tracker = (
            self.roi_annotation_fpath is not None
        )  # config["use_tracker"]
        self.is_enabled = True

        if self.roi_annotation_fpath is not None and not os.path.exists(
            self.roi_annotation_fpath
        ):
            print(f"ROI annotation file not found: {self.roi_annotation_fpath}")
            self.use_tracker = False
            self.is_enabled = False

        self.tracker = None
        if self.use_tracker:
            self.tracker = cv2.TrackerCSRT_create()
            with open(self.roi_annotation_fpath) as f:
                roi = pd.read_csv(f, delimiter=",")
            game_roi = roi[roi["game"] == self.game]
            if not game_roi.empty:
                self.init_roi = game_roi[["x", "y", "w", "h"]].values.flatten().tolist()
            else:
                print(f"ROI annotation not found for game {self.game}")
                self.use_tracker = False
                self.is_enabled = False

        self.env = make_retro(
            game=self.game,
            skip_frames=self.n_skip_frames,
            render_mode="rgb_array",
            valid_action_combos=self.valid_action_combos,
        )

    def get_name(self):
        return f"retro_{self.game.replace(' ', '_').lower()}"

    def get_info(self):
        return {
            "game": self.game,
            "config": self.config,
            "metadata": self.env.metadata,
            "action_space": self.env.action_space.shape,
            "observation_space": self.env.observation_space.shape,
            "action_space_type": self.env.unwrapped.use_restricted_actions.value,
            "action_captions": self.env.combos,
        }

    def generator(self, instance_id, session_id, n_steps_max):
        env = self.env
        ob, _ = env.reset()
        env.action_space.seed()
        np.random.seed()
        random.seed()

        box_size = 0
        if self.use_tracker:
            # scale the box to the player's height
            self.tracker.init(ob, self.init_roi)
            box_size = min(self.init_roi[3] * 5, ob.shape[0], ob.shape[1])

        action_noop = np.zeros_like(env.action_space.sample())
        for frame_id in range(self.n_skip_start_frames // self.n_skip_frames):
            ob, totrew, terminated, truncated, info = env.step(action_noop)
            session_end = frame_id == n_steps_max - 1 or terminated or truncated
            if session_end:
                return

        for frame_id in range(n_steps_max):
            action = env.action_space.sample()
            ob, totrew, terminated, truncated, info = env.step(action)
            frame = ob
            session_end = frame_id == n_steps_max - 1 or terminated or truncated
            extras = info
            tracking_background_mask = None

            if self.use_tracker:
                try:
                    frame, tracking_background_mask = apply_tracker(
                        self.tracker, frame, box_size
                    )
                except:
                    print(f"Failed to track object in frame {frame_id}")
                    break

            if self.image_size is not None:
                frame = pad_to_match_aspect_ratio(frame, self.image_size)
                frame = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)

            extras["tracking_background_mask"] = tracking_background_mask

            yield {
                "src_frame_id": frame_id - 1,
                "tgt_frame_id": frame_id,
                "frame": frame,
                "action": action.tolist(),
                "session_end": session_end,
                "extras": extras,
            }

            if session_end:
                break


class RetroActAutoExploreConnector(RetroActConnector):
    """Connector that uses the in-repo AutoExplore agent to act.

    Expects optional fields under self.config.get('agent', {}):
    - checkpoint_fpath: path to AutoExplore checkpoint (.pt) [preferred]
      (fallback: checkpoint_path for backward compatibility)
    - temperature: float, defaults to 1.0
    - gamma: float in [0,1], probability of taking random action (default 0.0)
    """

    def generator(self, instance_id, session_id, n_steps_max):
        # Lazy imports to avoid imposing deps when unused
        from pathlib import Path
        import torch
        import numpy as np

        # Scope shims to this connector only
        ensure_auto_explore_imports()

        # Import required classes via string for clarity
        ActorCritic = import_string("auto_explore.src.models.actor_critic:ActorCritic")
        AutoExploreAgent = import_string("auto_explore.src.agent:AutoExploreAgent")

        env = self.env
        observation, _ = env.reset()

        # Derive action dimension from the wrapped one-hot action space
        num_actions = getattr(env.action_space, "n", None)
        if num_actions is None:
            if self.valid_action_combos is not None:
                num_actions = len(self.valid_action_combos)
            else:
                raise ValueError("Cannot infer number of actions. Provide valid_action_combos in config.")

        agent_cfg = self.config.get("agent", {}) or {}
        ckpt_path = agent_cfg.get("checkpoint_fpath") or agent_cfg.get("checkpoint_path")
        if ckpt_path is None:
            raise ValueError("AutoExplore connector requires agent.checkpoint_fpath (or legacy checkpoint_path) in connector config.")
        temperature = float(agent_cfg.get("temperature", 1.0))
        gamma = float(agent_cfg.get("gamma", 0.0))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor_critic = ActorCritic(act_vocab_size=num_actions).to(device)
        # Optionally run the policy in bfloat16; keep FP32 on CPU for safety
        use_bf16 = device.type == "cuda"
        if use_bf16:
            actor_critic = actor_critic.to(dtype=torch.bfloat16)
        # Initialize recurrent state for a single environment stream
        actor_critic.reset(n=1)
        agent = AutoExploreAgent(world_model=None, actor_critic=actor_critic).to(device)
        agent.load(path_to_checkpoint=Path(ckpt_path), device=device, load_tokenizer=False, load_world_model=False, load_actor_critic=True)
        agent.eval()

        # Respect initial skip frames as in base connector
        action_noop = np.zeros_like(env.action_space.sample())
        for frame_id in range(self.n_skip_start_frames // self.n_skip_frames):
            observation, totrew, terminated, truncated, info = env.step(action_noop)
            session_end = frame_id == n_steps_max - 1 or terminated or truncated
            if session_end:
                return

        # Optional tracker init
        if self.use_tracker:
            self.tracker.init(observation, getattr(self, "init_roi", None))
            box_size = min(self.init_roi[3] * 5, observation.shape[0], observation.shape[1])
        else:
            box_size = 0

        for frame_id in range(n_steps_max):
            # Prepare observation tensor (1, 3, H, W) in [0,1]
            obs_t = torch.from_numpy(observation).to(device)
            if obs_t.dtype != torch.float32:
                obs_t = obs_t.float()
            obs_t = (obs_t / 255.0).permute(2, 0, 1).unsqueeze(0)
            if use_bf16:
                obs_t = obs_t.to(torch.bfloat16)

            with torch.no_grad():
                act_token = agent.act(obs_t, should_sample=False, temperature=temperature)
                act_idx = int(act_token.item())

            # Optional exploration
            if gamma > 0.0 and random.random() < gamma:
                act_idx = random.randrange(num_actions)

            action = np.zeros(num_actions, dtype=np.uint8)
            action[act_idx] = 1

            observation, reward, terminated, truncated, info = env.step(action)
            frame = observation
            session_end = frame_id == n_steps_max - 1 or terminated or truncated

            extras = dict(info)
            tracking_background_mask = None
            if self.use_tracker:
                try:
                    frame, tracking_background_mask = apply_tracker(self.tracker, frame, box_size)
                except Exception:
                    # If tracking fails, end this session gracefully
                    break

            if self.image_size is not None:
                frame = pad_to_match_aspect_ratio(frame, self.image_size)
                frame = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)

            extras["tracking_background_mask"] = tracking_background_mask
            extras["temperature"] = temperature
            extras["gamma"] = gamma

            yield {
                "src_frame_id": frame_id - 1,
                "tgt_frame_id": frame_id,
                "frame": frame,
                "action": action.tolist(),
                "session_end": session_end,
                "extras": extras,
            }

            if session_end:
                break


class RetroActAgent57Connector(RetroActConnector):
    """Connector that uses a Deep RL Zoo Agent57 actor to act.

    Expects fields under self.config.get('agent', {}):
    - checkpoint_fpath: path to Agent57 checkpoint (.ckpt) [preferred]
      (fallback: checkpoint_path for backward compatibility)
    - gamma: float in [0,1], probability of taking random action (default 0.0)
    Note: action head size must match len(valid_action_combos) in most checkpoints.
    """

    def generator(self, instance_id, session_id, n_steps_max):
        import numpy as np
        import cv2
        import random

        # Lazy import heavy dependencies; raise actionable error if missing
        try:
            ensure_deep_rl_zoo_imports()
            from deep_rl_zoo import greedy_actors
            from deep_rl_zoo.networks.value import Agent57Conv2dNet
            from deep_rl_zoo.networks.curiosity import RndConvNet, NguEmbeddingConvNet
            import deep_rl_zoo.types as types_lib
            from deep_rl_zoo.checkpoint import PyTorchCheckpoint
            import torch
        except Exception as e:
            raise ImportError("Agent57 dependencies not found. Install deep_rl_zoo and its deps or use variant=random/auto_explore.") from e

        env = self.env
        observation, _ = env.reset()

        num_actions = getattr(env.action_space, "n", None)
        if num_actions is None:
            if self.valid_action_combos is not None:
                num_actions = len(self.valid_action_combos)
            else:
                raise ValueError("Cannot infer number of actions. Provide valid_action_combos in config.")

        agent_cfg = self.config.get("agent", {}) or {}
        ckpt_path = agent_cfg.get("checkpoint_fpath") or agent_cfg.get("checkpoint_path")
        if ckpt_path is None:
            raise ValueError("Agent57 connector requires agent.checkpoint_fpath (or legacy checkpoint_path) in connector config.")
        gamma = float(agent_cfg.get("gamma", 0.0))

        # Build networks and load checkpoint
        state_dim = (1, 84, 84)
        action_dim = num_actions
        runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        network = Agent57Conv2dNet(state_dim=state_dim, action_dim=action_dim, num_policies=32)
        rnd_target_network = RndConvNet(state_dim=state_dim)
        rnd_predictor_network = RndConvNet(state_dim=state_dim)
        embedding_network = NguEmbeddingConvNet(state_dim=state_dim, action_dim=action_dim)

        checkpoint = PyTorchCheckpoint(environment_name=self.game, agent_name="Agent57", restore_only=True)
        checkpoint.register_pair(("network", network))
        checkpoint.register_pair(("rnd_target_network", rnd_target_network))
        checkpoint.register_pair(("rnd_predictor_network", rnd_predictor_network))
        checkpoint.register_pair(("embedding_network", embedding_network))
        checkpoint.restore(ckpt_path)

        network.eval()
        rnd_target_network.eval()
        rnd_predictor_network.eval()
        embedding_network.eval()

        agent = greedy_actors.Agent57EpsilonGreedyActor(
            network=network,
            embedding_network=embedding_network,
            rnd_target_network=rnd_target_network,
            rnd_predictor_network=rnd_predictor_network,
            exploration_epsilon=0.01,
            episodic_memory_capacity=5000,
            num_neighbors=10,
            kernel_epsilon=0.0001,
            cluster_distance=0.008,
            max_similarity=8.0,
            random_state=np.random.RandomState(1),
            device=runtime_device,
        )

        agent.reset()

        # Warm-up skip frames with NOOP
        action_noop = np.zeros_like(env.action_space.sample())
        for frame_id in range(self.n_skip_start_frames // self.n_skip_frames):
            observation, totrew, terminated, truncated, info = env.step(action_noop)
            session_end = frame_id == n_steps_max - 1 or terminated or truncated
            if session_end:
                return

        reward = 0.0
        done = False
        loss_life = False
        first_step = True
        info = {}

        # Optional tracker init
        if self.use_tracker:
            self.tracker.init(observation, getattr(self, "init_roi", None))
            box_size = min(self.init_roi[3] * 5, observation.shape[0], observation.shape[1])
        else:
            box_size = 0

        for frame_id in range(n_steps_max):
            # Prepare 84x84 grayscale observation for the agent
            ob_gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            ob_gray = cv2.resize(ob_gray, (84, 84), interpolation=cv2.INTER_AREA)[None, ...]

            timestep_t = types_lib.TimeStep(
                observation=ob_gray,
                reward=reward,
                done=done or loss_life,
                first=first_step,
                info=info,
            )
            a_t = agent.step(timestep_t)

            if gamma > 0.0 and random.random() < gamma:
                a_t = random.randrange(action_dim)

            action = np.eye(action_dim, dtype=np.uint8)[a_t]

            observation, reward, done, truncated, info = env.step(action)
            frame = observation
            session_end = frame_id == n_steps_max - 1 or done or truncated

            extras = dict(info)
            tracking_background_mask = None
            if self.use_tracker:
                try:
                    frame, tracking_background_mask = apply_tracker(self.tracker, frame, box_size)
                except Exception:
                    break

            if self.image_size is not None:
                frame = pad_to_match_aspect_ratio(frame, self.image_size)
                frame = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)

            extras["tracking_background_mask"] = tracking_background_mask
            extras["gamma"] = gamma

            yield {
                "src_frame_id": frame_id - 1,
                "tgt_frame_id": frame_id,
                "frame": frame,
                "action": action.tolist(),
                "session_end": session_end,
                "extras": extras,
            }

            first_step = False
            if session_end:
                # Flush final done state for the agent57 actor, if needed
                if done:
                    ob_gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
                    ob_gray = cv2.resize(ob_gray, (84, 84), interpolation=cv2.INTER_AREA)[None, ...]
                    timestep_t = types_lib.TimeStep(
                        observation=ob_gray,
                        reward=reward,
                        done=True,
                        first=False,
                        info=info,
                    )
                    _ = agent.step(timestep_t)  # noqa: F841
                break


def run_game(selected_game, args):
    try:
        env = make_retro(
            game=selected_game,
            state=args.state,
            scenario=args.scenario,
            skip_frames=args.skip_frames,
        )
    except FileNotFoundError as e:
        print(e)
        exit(0)
    env.reset()

    while True:
        sample = env.action_space.sample()
        ob, totrew, terminated, truncated, info = env.step(sample)
        print(info)
        if terminated or truncated:
            env.reset()

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Airstriker-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--max_steps", default=None)
    parser.add_argument("--skip_frames", default=1)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    games_list = retro.data.list_games(inttype=retro.data.Integrations.ALL)
    selected_games = games_list

    pool = multiprocessing.Pool(1)

    results = []
    for game in selected_games:
        pool.apply_async(run_game, args=(game, args))
    pool.close()
    pool.join()
    # run_game(selected_game, args)
    # env.render()
    # def make_env():
    #     env = make_retro(game=args.game, state=args.state, scenario=args.scenario)
    #     env = wrap_deepmind_retro(env)
    #     return env

    # venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 8), n_stack=4))
    # model = PPO(
    #     policy="CnnPolicy",
    #     env=venv,
    #     learning_rate=lambda f: f * 2.5e-4,
    #     n_steps=128,
    #     batch_size=32,
    #     n_epochs=4,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.1,
    #     ent_coef=0.01,
    #     verbose=1,
    # )
    # model.learn(
    #     total_timesteps=100_000_000,
    #     log_interval=1,
    # )


if __name__ == "__main__":
    main()
