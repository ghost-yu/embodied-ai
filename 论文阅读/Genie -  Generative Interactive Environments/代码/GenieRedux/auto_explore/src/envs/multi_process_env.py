from dataclasses import astuple, dataclass
from enum import Enum
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Callable, Iterator, List, Optional, Tuple
from lovely_numpy import lo
import numpy as np
from einops import rearrange

from tools.logger import getLogger
log = getLogger(__name__)

from .done_tracker import DoneTrackerEnv
from .retrowrapper import RetroWrapper, set_retro_make
import pickle


class MessageType(Enum):
    RESET = 0
    RESET_RETURN = 1
    STEP = 2
    STEP_RETURN = 3
    CLOSE = 4


@dataclass
class Message:
    type: MessageType
    content: Optional[Any] = None

    def __iter__(self) -> Iterator:
        return iter(astuple(self))


def child_env(child_id: int, env_fn: Callable, child_conn: Connection) -> None:
    np.random.seed(child_id + np.random.randint(0, 2 ** 31 - 1))
    env = env_fn()
    while True:
        message_type, content = child_conn.recv()
        if message_type == MessageType.RESET:
            obs = env.reset()
            child_conn.send(Message(MessageType.RESET_RETURN, obs))
        elif message_type == MessageType.STEP:
            obs, rew, terminated, truncated, _ = env.step(content)
            done = terminated or truncated
            if terminated or truncated:
                obs = env.reset()
            child_conn.send(Message(MessageType.STEP_RETURN, (obs, rew, done, None)))
        elif message_type == MessageType.CLOSE:
            child_conn.close()
            return
        else:
            raise NotImplementedError

def process_obs_np(obs, transform):
    new_obs = []
    for ob in obs:
        ob = transform(ob)
        new_obs.append(ob)
    return new_obs

class MultiProcessEnv(DoneTrackerEnv):
    def __init__(self, env_fn: Callable, num_envs: int, should_wait_num_envs_ratio: float, transform: int|None =None) -> None:
        super().__init__(num_envs)
        self.transform = transform

        
        self.num_actions = env_fn().action_space.n
        # self.num_actions = env_fn().env.action_space.n
        self.should_wait_num_envs_ratio = should_wait_num_envs_ratio
        self.processes, self.parent_conns = [], []
        for child_id in range(num_envs):
            parent_conn, child_conn = Pipe()
            self.parent_conns.append(parent_conn)
            p = Process(target=child_env, args=(child_id, env_fn, child_conn), daemon=True)
            self.processes.append(p)
        for p in self.processes:
            p.start()

    def should_reset(self) -> bool:
        return (self.num_envs_done / self.num_envs) >= self.should_wait_num_envs_ratio

    def _receive(self, check_type: Optional[MessageType] = None) -> List[Any]:
        messages = [parent_conn.recv() for parent_conn in self.parent_conns]
        if check_type is not None:
            assert all([m.type == check_type for m in messages])
        return [m.content for m in messages]

    def reset(self) -> np.ndarray:
        self.reset_done_tracker()
        for parent_conn in self.parent_conns:
            parent_conn.send(Message(MessageType.RESET))
        content = self._receive(check_type=MessageType.RESET_RETURN)
        content = [c[0] for c in content]
        if self.transform is not None:
            for idx in range(len(content)):
                temp = np.expand_dims(np.array(content[idx]), axis=0)
                temp = process_obs_np(temp, self.transform)
                content[idx] = rearrange(temp, 't c h w -> t h w c')[0]
        return np.stack(content)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(Message(MessageType.STEP, action))
        content = self._receive(check_type=MessageType.STEP_RETURN)
        obs, rew, done, _ = zip(*content)
        if isinstance(obs, tuple):
            obs = list(obs)
        for idx in range(len(obs)):
            if isinstance(obs[idx], tuple):
                obs[idx] = obs[idx][0]
        done = np.stack(done)
        self.update_done_tracker(done)
        
        new_obs = []
        if self.transform is not None:
            for idx in range(len(obs)):
                try:
                    temp = np.expand_dims(np.array(obs[idx]), axis=0)
                    temp = process_obs_np(temp, self.transform)
                    temp = rearrange(temp, 't c h w -> t h w c')[0]
                    new_obs.append(temp)
                except Exception as e:
                    log.e("Wut", idx, obs)
                    #save obs[idx] to file
                    with open(f'/home/nedko_savov/projects/ivg/external/open-genie/obs_{idx}.pkl', 'wb') as f:
                        pickle.dump(obs[idx], f)
                    for o in obs[idx]:
                        log.e("W", len(o))
                        log.e("H", len(o[0]))
                        
                    log.e(f"Error in step: {e}")
                    raise e
        else:
            log.e("No transform")
            new_obs = obs

        try:
            new_obs = np.stack(new_obs)
        except Exception as e:
            log.e("Stack fail", new_obs)
            
            raise Exception(e)
        return new_obs, np.stack(rew), done, None

    def close(self) -> None:
        for parent_conn in self.parent_conns:
            parent_conn.send(Message(MessageType.CLOSE))
        for p in self.processes:
            p.join()
        for parent_conn in self.parent_conns:
            parent_conn.close()
