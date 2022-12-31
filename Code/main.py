
import threading
import numpy as np
import os
from torch._C import device

import gym
import ma_gym



import numpy as np
import random
import torch
from torch import optim
import threading
from tqdm import tqdm
#from lib.common_ptan.agent import MADDPGAgent
import  settings
from agent import DQNAgent_ind
from networks import HCritic, Critc
from actions import ArgmaxActionSelector, EpsilonGreedyActionSelector
import experience
from utils import test_env, unpack_batch, history, history_ind

def moving_average(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')

    
from tensorboardX import SummaryWriter, writer

if __name__ == "__main__":
    agent_num = 2

    state_dim = 2
    action_dim = 5



    args = settings.HYPERPARAMS['Switch4-v0']

    args.n_agents = agent_num

    args.obs_shape = [state_dim]*agent_num
    args.action_shape = [action_dim]*agent_num

    sa_sizes = []
    for i in range(agent_num):
        sa_sizes.append((state_dim, action_dim))

    target_update_steps = 10

    env = gym.make(args.scenario_name)

    obs = env.reset()


    # agent initialisation
    writer = SummaryWriter()



    critic = [Critc(args, i) for i in range(agent_num)]
    action_selector = EpsilonGreedyActionSelector(epsilon=0.1)

    agent = DQNAgent_ind(args, critic, action_selector)
    source = experience.MOARLExperienceSource(env, agent, args, steps_count = 1) #load the experience source
    replay = experience.ExperienceReplayBuffer(source, args.buffer_size) #load the replay buffer to store the episodes

    for time_step in tqdm(range(0, args.time_steps, args.n_threads)): #iterate through the fixed number of episodes
        #print("time_steps %i-%i of %i" % (time_step + 1,time_step + 1 + args.n_threads,args.time_steps))

        args.train = False

        replay.populate(1) #gather experiences/populate the buffer

        
        replay.populate(1) #gather experiences/populate the buffer
        if len(replay.buffer) < 2*args.batch_size: #continue further only if the buffer is filled with the twice of the batch size
            continue

        batch_ = replay.sample(args.batch_size) #sample the batches from source

        

        Multi_batch_sample = unpack_batch(batch_, args)

        
        

        loss = 0
        for i in range(args.n_agents):
            sample = history_ind(args, Multi_batch_sample, i)
            c_loss = agent.train(sample, i, writer=writer)
            agent.step_update(c_loss, i)
            loss += c_loss
        writer.add_scalar('loss', loss, time_step)

        #for i in range(args.n_agents):
            
        
        if time_step % args.target_update == 0:
        
            for i in range(args.n_agents):
                agent.update_target(i)
        
        

        #MARL_utils.save_results(args, env, agent, time_step, writer)
        test_env(args, env, agent, time_step, writer)
        
    writer.close()

    
    