import numpy as np
from coinrun import setup_utils, make
import argparse
import cv2
import tqdm

import coinrun.policies as policies
import coinrun.main_utils as utils


def random_agent(num_envs=1, max_steps=100000):
    setup_utils.setup_and_load(use_cmd_line_args=False, num_levels=10000)
    env = make('standard', num_envs=num_envs)
    for step in range(max_steps):
        acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        _obs, rews, _dones, _infos = env.step(acts)
        # yield _obs, acts, rews, _dones, _infos
        print(_obs.shape, acts.shape)
        #print the mean and variance of the observations
        print("mean", np.mean(_obs))
        print("step", step, "rews", rews)
    env.close()

def random_agent_generator(num_envs=1, max_steps=100000, is_high_difficulty=True, is_high_res=True, should_paint_velocity=False, seed_ids = [0]):

    setup_utils.setup_and_load(use_cmd_line_args=False, num_levels=10000, high_difficulty=is_high_difficulty, is_high_res=is_high_res, paint_vel_info=should_paint_velocity, set_seed=seed_ids[0])
    env = make('standard', num_envs=num_envs)
    # from coinrun.config import Config
    # Config.IS_HIGH_RES = is_high_res
    
    
    for step in tqdm.tqdm(range(max_steps)):
        prev_acts = None
        while True:
            acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
            # if step % 2 ==0:
            #     acts = np.array([6 for _ in range(env.num_envs)])
            # else:
            #     acts = np.array([0 for _ in range(env.num_envs)])
            
            if prev_acts is None or not np.array_equal(acts, prev_acts):
                break
        prev_acts = acts
        
        env.hires_render = is_high_res
        _obs, rews, _dones, _infos = env.step(acts)
        env.hires_render = False
        # obs = env.get_images()
        # extras = {"extra_obs": obs}
        extras = {}

        yield _obs, acts, rews, _dones, _infos, extras
    env.close()

def create_act_model(sess, env, nenvs):
    ob_space = env.observation_space
    ac_space = env.action_space

    policy = policies.get_policy()
    act = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)

    return act

def ppo_init(is_high_difficulty=True, is_high_res=True, should_paint_velocity=False, seed_ids = [0]):
    from coinrun.config import Config
    Config.WORKDIR = './external/coinrun/coinrun/saved_models'
    setup_utils.setup_and_load(restore_id="myrun", use_cmd_line_args=False, num_levels=10000, high_difficulty=is_high_difficulty, is_high_res=is_high_res, paint_vel_info=should_paint_velocity, set_seed=seed_ids[0])

def ppo_agent_generator(num_envs=1, max_steps=100000, is_high_difficulty=True, is_high_res=True, should_paint_velocity=False, seed_ids = [0]):
    
    import tensorflow as tf
    ppo_init(is_high_difficulty=is_high_difficulty, is_high_res=is_high_res, should_paint_velocity=should_paint_velocity, seed_ids=seed_ids)
    # setup_utils.setup_and_load(restore_id="myrun", use_cmd_line_args=False, num_levels=10000, high_difficulty=is_high_difficulty, is_high_res=is_high_res, paint_vel_info=should_paint_velocity, set_seed=seed_ids[0])
    env = make('standard', num_envs=num_envs)
    _obs = env.reset()
    
    _dones = np.zeros(num_envs)
    
    with tf.Session() as sess:
        agent = create_act_model(sess, env, num_envs)    
        sess.run(tf.global_variables_initializer())
        loaded_params = utils.load_params_for_scope(sess, 'model')

        state = agent.initial_state
        if not loaded_params:
            print('NO SAVED PARAMS LOADED')

        for step in tqdm.tqdm(range(max_steps)):
            #reshape the observation tensor to shape '(1, 64, 64, 3)'
            _obs = np.array([cv2.resize(img, (64, 64)) for img in _obs])
            acts, values, state, _ = agent.step(_obs, state, _dones)
            # acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
            
            env.hires_render = True
            _obs, rews, _dones, _infos = env.step(acts)
            env.hires_render = False
            obs = env.get_images()
            extras = {"extra_obs": obs}

            yield _obs, acts, rews, _dones, _infos, extras
            #print the mean and variance of the observations
    env.close()

if __name__ == '__main__':
    #add arguments for max steps and num envs
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--num_envs', type=int, default=1)
    args = parser.parse_args()
    random_agent(num_envs=args.num_envs, max_steps=args.max_steps)