from pathlib import Path

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from models.genie_redux import GenieReduxGuided

from .models.actor_critic import ActorCritic
from .utils import extract_state_dict



class AutoExploreAgent(nn.Module):
    def __init__(self, world_model: GenieReduxGuided, actor_critic: ActorCritic):
        super().__init__()
        self.world_model = world_model
        self.actor_critic = actor_critic

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        # Tokenizer state is no longer used/loaded
        if load_actor_critic:
            self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, 'actor_critic'))

    @torch.no_grad()
    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        input_ac = obs
        input_ac = nn.functional.interpolate(input_ac, size=(64, 64), mode='bilinear', align_corners=False, antialias=True)
        logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token
