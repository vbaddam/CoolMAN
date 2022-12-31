
from sys import winver
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
from agent import MAPPO
from networks import Actor_p, Critic_p
from actions import ArgmaxActionSelector, EpsilonGreedyActionSelector
import buff
from utils import test_env, unpack_batch, history, history_ind
from menv import make_env
torch.autograd.set_detect_anomaly(True)



from tensorboardX import SummaryWriter, writer


if __name__ == "__main__":
    # Hyperparameters
    args = settings.HYPERPARAMS['Switch4-v0'] #Get arguments for the given environment

    agent_num = 2
    state_dim = 2
    action_dim = 5


    args.multi_discrete = False
    args.discrete = True
    args.continuous = False

    args.n_agents = agent_num

    args.obs_shape = [state_dim]*agent_num
    args.action_shape = [action_dim]*agent_num
    args.action_space = ['Discrete']*agent_num

    target_update_steps = 10

    env = gym.make(args.scenario_name)

    obs = env.reset()


    # agent initialisation
    writer = SummaryWriter()


    


    device = torch.device("cuda" if args.cuda else "cpu")

    writer = SummaryWriter('logs/'+args.scenario_name)

    a_net = [Actor_p(args, i) for i in range(args.n_agents)]
    c_net = [Critic_p(args, i) for i in range(args.n_agents)]

    

    args.training = True

    if args.training:

        Agent = MAPPO(args, a_net, c_net, device)
        source = buff.MOARLExperienceSource(env, Agent, args, steps_count = 1) #load the experience source
        
        replay = buff.ExperienceReplayBuffer(source, args.buffer_size)

        for time_step in tqdm(range(0, args.time_steps)): #iterate through the fixed number of episodes
            #print("time_steps %i-%i of %i" % (time_step + 1,time_step + 1 + args.n_threads,args.time_steps))
        
            replay.populate(1) #gather experiences/populate the buffer
            if len(replay.buffer)  == args.buffer_size: #continue further only if the buffer is filled with the twice of the batch size
            
                
                rew = source.pop_total_rewards()
                
                trajectory = replay.sample(args.batch_size)

                Agent.train(trajectory, writer) 

                
                #writer.add_scalar('avergae_return', rew, time_step)
                
                replay.clear()

                
                test_env(args, env, Agent, writer, time_step)
    
    writer.close()
    
