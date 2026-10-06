from generator.connector_base import BaseConnector
from coinrun.random_agent import (
    ppo_agent_generator,
    random_agent_generator,
    ppo_init,
)
import cv2

class CoinRunConnector(BaseConnector):
    def __init__(self, config=None):
        if config is None:
            config = {
                "name": "coinrun",
                "version": "0.1.0",
                "is_high_res": False,
                "is_high_difficulty": True,
                "should_paint_velocity": False,
                "agent_type": "ppo",
            }

        self.config = config
        self.name = config["name"]
        self.version = config["version"]
        self.is_high_res = config["is_high_res"]
        self.is_high_difficulty = config["is_high_difficulty"]
        self.should_paint_velocity = config["should_paint_velocity"]
        self.image_size = config["image_size"]
        self.agent_type = config["agent_type"]

        self.agent_generator = (
            ppo_agent_generator if self.agent_type == "ppo" else random_agent_generator
        )

    def get_name(self):
        return "coinrun"

    def get_info(self):
        return {
            "action_space": [7],
            "observation_space": [512, 512] if self.is_high_res else [256, 256],
            "config": self.config,
        }

    def generator(self, instance_id, session_id, n_steps_max):

        for frame_id, (_obs, acts, rews, _dones, _infos, extras) in enumerate(
            self.agent_generator(
                num_envs=1,
                max_steps=n_steps_max,
                is_high_difficulty=self.is_high_difficulty,
                is_high_res=self.is_high_res,
                should_paint_velocity=self.should_paint_velocity,
                seed_ids=[instance_id],
            )
        ):
            frame = _obs[0]
            action = acts[0]
            session_end = frame_id == n_steps_max - 1
            if self.image_size is not None:
                frame = cv2.resize(frame, self.image_size)

            if _dones[0]:
                break

            yield {
                "src_frame_id": frame_id - 1,
                "tgt_frame_id": frame_id,
                "frame": frame,
                "action": int(action),
                "session_end": session_end,
                "extras": extras,
            }
            if rews[0] > 0:
                break
