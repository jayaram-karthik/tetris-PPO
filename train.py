import ppo
import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from env import TetrisEnv

from ppo import PPO

def train():
    env_name = "Tetris_env"

    max_ep_len = 1000
    max_training_timesteps = int(3e6)

    print_freq = max_ep_len * 10
    save_model_freq = int(1e5)

    update_timestep = max_ep_len * 4
    policy_update = 80

    eps_clip = 0.2
    gamma = 0.999

    lr_actor = 3e-4
    lr_critic = 1e-3

    random_seed = 0 

    print("training environment name : " + env_name)

    env = TetrisEnv()

    # state space dimension
    state_dim = env.board.shape

    # action space dimension
    action_dim = 7

    run_num_pretrained = 0

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, policy_update, eps_clip)


    print("============================================================================================")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 0
    
    checkpoint_path = directory + f"PPO_{env_name}_{random_seed}_{run_num_pretrained}.pth"

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    env.close()