from nn import MLP
from envs import HalfCheetahDirEnv
from utils import ReplayBuffer
import hydra
from hydra.utils import get_original_cwd
import json
from collections import namedtuple
import pickle
import torch
import torch.optim as O
from typing import List
import higher
from itertools import count
import logging
from utils import Experience
from losses import policy_loss_on_batch, vf_loss_on_batch, qf_loss_on_batch
from impl import build_networks_and_buffers, get_opts_and_lrs, get_env
import gym
import random
import math

LOG = logging.getLogger(__name__)

EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 2000


def rollout_policy(policy: MLP, env, render: bool = False) -> List[Experience]:
    trajectory = []
    
    state = env.reset()

    if render:
        env.render()
    done = False
    total_reward = 0
    episode_t = 0
    success = False
    policy.eval()
    current_device = list(policy.parameters())[-1].device
    while not done:
        with torch.no_grad():
            action = policy(torch.tensor(state).to(current_device).float()).squeeze()

            np_action = action.squeeze().cpu().numpy()
            np_action = np_action.clip(min=env.action_space.low, max=env.action_space.high)

            
            
        next_state, reward, done, info_dict = env.step(np_action)

        if "success" in info_dict and info_dict["success"]:
            success = True

        if render:
            env.render()
            
        trajectory.append(Experience(state, np_action, next_state, reward, done))
        state = next_state
        total_reward += reward
        episode_t += 1
        if episode_t >= env._max_episode_steps or done:
            break
    
    return trajectory, total_reward, success
    
def generate_episode(env, policy, num_episodes=1, random=False):
    
    trajectories = []
    current_device = list(policy.parameters())[-1].device

    for i in range(num_episodes):
        state = env.reset()
        done = False
        trajectory = []
        
        episode_t = 0
        while not done:
            if random:
                np_action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy(torch.tensor(state).to(current_device).float()).squeeze()

                    np_action = action.squeeze().cpu().numpy()
                    np_action = np_action.clip(min=env.action_space.low, max=env.action_space.high)
                
            next_state, reward, done, info_dict = env.step(np_action)
            reward = -1 * reward

            trajectory.append(Experience(state, np_action, next_state, reward, done))
            episode_t += 1
            if episode_t >= env._max_episode_steps or done:
                break
    
        trajectories.append(trajectory)
        
    return trajectories

def select_action(state, time, env, policy):
    sample = random.random()
    
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * time / EPS_DECAY)
    
    # print(eps_threshold)
    
    if sample > eps_threshold:
        current_device = list(policy.parameters())[-1].device

        with torch.no_grad():
                action = policy(torch.tensor(state).to(current_device).float()).squeeze()

                np_action = action.squeeze().cpu().numpy()
                np_action = np_action.clip(min=env.action_space.low, max=env.action_space.high)
                np_action = env.action_space.sample()
    else:
        np_action = env.action_space.sample()
    
    return np_action

@hydra.main(config_path="config", config_name="config.yaml")
def main(args):
    if args.colab:
        with open(f"{get_original_cwd()}/{args.colab_task_config}", "r") as f:
            task_config = json.load(
                f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
            )
    else:
        with open(f"{get_original_cwd()}/{args.task_config}", "r") as f:
            task_config = json.load(
                f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
            )
    
    env = get_env(args, task_config, t=True)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    task_buffer = ReplayBuffer(
        args.inner_buffer_size,
        obs_dim,
        action_dim,
        discount_factor=0.99,
    )

    # warm-up
        
    policy, vf, task_buffers, q_function = build_networks_and_buffers(args, env, task_config, is_train=False)
    
    policy.load_state_dict(torch.load(task_config.policy))
    vf.load_state_dict(torch.load(task_config.vf))
    q_function.load_state_dict(torch.load(task_config.qf))
    
    policy_opt, vf_opt, qf_opt, policy_lrs, vf_lrs, qf_lrs = get_opts_and_lrs(args, policy, vf, q_function)


    trajectories = generate_episode(env=env, policy=policy, num_episodes=256, random=True)
    task_buffer.add_trajectories(trajectories)
    
    time = 0
    
    
    for i in range(20000):
        state = env.reset()
        done = False
        trajectory=[]
        episode_t = 0
        while not done:
            
            # Select action
            
            action = select_action(state, time, env, policy)
            time += 1
            
            next_state, reward, done, info_dict = env.step(action)
            reward = -1 * reward
            trajectory.append(Experience(state, action, next_state, reward, done))
            
            inner_batch = task_buffer.sample(
                args.inner_batch_size, return_dict=True, device=args.device
            )
            
            # Optimize

            loss = qf_loss_on_batch(q_function, inner_batch, inner=True)
            loss.backward()
            qf_opt.step()
            qf_opt.zero_grad()
            
            loss = vf_loss_on_batch(vf, q_function, inner_batch, inner=True)
            loss.backward()
            vf_opt.step()
            vf_opt.zero_grad()
            
            loss = policy_loss_on_batch(
                policy, 
                vf,
                q_function,
                inner_batch,
                args.advantage_head_coef, inner=True
            )        
            
            loss.backward()
            policy_opt.step()
            policy_opt.zero_grad()
            
            episode_t += 1
            if episode_t >= env._max_episode_steps or done:
                task_buffer.add_trajectory(trajectory)
                break
            
        
        
        if i % args.rollout_interval == 0:
            adapted_trajectory, adapted_reward, success = rollout_policy(policy, env)
            LOG.info(f"Task {i} reward: {adapted_reward}")
                
        
if __name__ == '__main__':
    main()