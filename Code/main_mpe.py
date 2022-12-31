
import threading
import numpy as np
import os
from torch._C import device

import gym




import numpy as np
import random
import torch
from torch import optim
import threading
from tqdm import tqdm
#from lib.common_ptan.agent import MADDPGAgent
import  settings
import experience
from menv import make_env
from networks import Actor_p, HCritic, AttentionCritic
from agent import MAPPO
from utils import test_env_mpe





#from torch.utils.tensorboard import SummaryWriterA
from tensorboardX import SummaryWriter, writer

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
torch.autograd.set_detect_anomaly(True)



if __name__ == '__main__':


    args = settings.HYPERPARAMS['simple_spread'] #Get arguments for the given environment

    env = make_env(args)

    obs = env.reset()
    

    agent_init_params = []
    sa_size = []
    for acsp, obsp in zip(env.action_space,
                            env.observation_space):
        agent_init_params.append({'num_in_pol': obsp.shape[0],
                                    'num_out_pol': acsp.n})
        sa_size.append((obsp.shape[0], acsp.n))

    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    args.action_shape = action_shape[:args.n_agents]
    print(args.action_shape)
    args.high_action = 1
    args.low_action = -1


    args.discrete = True
    args.continuous = False
    args.one_hot = True

    args.multi_discrete = False
    args.action_space = "Discrete"
    args.action_space = [["Discrete"]*args.n_agents][0]

    args.obs_global_shape = [len(obs[0])]
    print('the length of global', args.obs_global_shape)
    

    device = torch.device("cuda" if args.cuda else "cpu")

    writer = SummaryWriter('logs/'+args.scenario_name)

    a_net = [Actor_p(args, i) for i in range(args.n_agents)]
    c_net = AttentionCritic(args, sa_size)

    args.training = True

    if args.training:

        Agent = MAPPO(args, a_net, c_net, device)
        source = experience.MOARLExperienceSource(env, Agent, args, steps_count = 1) #load the experience source
        
        replay = experience.ExperienceReplayBuffer(source, args.buffer_size)

        for time_step in tqdm(range(0, args.time_steps)): #iterate through the fixed number of episodes
            #print("time_steps %i-%i of %i" % (time_step + 1,time_step + 1 + args.n_threads,args.time_steps))
        
            replay.populate(1) #gather experiences/populate the buffer
            if len(replay.buffer)  == args.buffer_size: #continue further only if the buffer is filled with the twice of the batch size
            
                
                rew = source.pop_total_rewards()
                
                trajectory = replay.sample(args.batch_size)

                #Agent.train(trajectory, writer) 
                advantages = Agent.update_critic(trajectory, writer)
                Agent.update_actor(trajectory, advantages, writer)

                
                #writer.add_scalar('avergae_return', rew, time_step)
                
                replay.clear()

                
                test_env_mpe(args, env, Agent, writer, time_step)
    
    else:
        args.use_load_model = True
        eval_episodes = 10
        Agent = MAPPO(args, a_net, c_net, device)
        for time_step in range(eval_episodes):
            test_env_mpe(args, env, Agent, writer, time_step)
    
    writer.close()