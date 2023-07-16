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
    

@hydra.main(config_path="config", config_name="config.yaml")
def main(args):
    with open(f"{get_original_cwd()}/{args.task_config}", "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )
    
    env = get_env(args, task_config)
    policy, vf, task_buffers, q_function = build_networks_and_buffers(args, env, task_config, is_train=False)
    
    policy.load_state_dict(torch.load('/home/arya/arya/dl/macaw-min/models/policy.pt'))
    vf.load_state_dict(torch.load('/home/arya/arya/dl/macaw-min/models/vf.pt'))
    q_function.load_state_dict(torch.load('/home/arya/arya/dl/macaw-min/models/qf.pt'))
    
    policy_opt, vf_opt, qf_opt, policy_lrs, vf_lrs, qf_lrs = get_opts_and_lrs(args, policy, vf, q_function)

    for i in range(500):
        inner_batch = task_buffers[0].sample(
                args.inner_batch_size, return_dict=True, device=args.device
            )
        
        loss = qf_loss_on_batch(q_function, inner_batch, inner=True)
        loss.backward()
        qf_opt.step()
        qf_opt.zero_grad()
        
        loss = vf_loss_on_batch(vf, q_function, inner_batch, inner=True)
        loss.backward()

        vf_opt.step()
        vf_opt.zero_grad()
        
        loss = policy_loss_on_batch(policy, vf,
                    q_function,
                    inner_batch,
                    args.advantage_head_coef, inner=True)
        loss.backward()

        policy_opt.step()
        policy_opt.zero_grad()
        
        
    
    rollout_policy(policy, env, render=True)

if __name__ == '__main__':
    main()