import gym
import time
import sys 
import os 

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal

from torch.autograd import Variable
from torch.distributions import Categorical
from utils import hard_update


sys.path.append('../../')
#from utilities.MARL_utils import MADDPG


class MAPPO:

    def __init__(self, args, Actor, Critic, device):

        self.args = args
        
        self.actor = Actor
        self.critic = Critic
        self.target_critic = Critic

        hard_update(self.target_critic, self.critic)

        self.device = device

        self.actor_optimizer = [torch.optim.Adam(self.actor[i].parameters(), lr=self.args.lr_actor) for i in range(args.n_agents)]
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic) 
        self.niter = 0

        self.model_path = os.path.join(args.save_dir, args.scenario_name)

        print(self.model_path)

        if args.use_load_model and os.path.exists(self.model_path):
            self.model_path = os.path.join(args.save_dir, args.scenario_name)
            self.model_path = os.path.join(self.model_path, '%d_agents' % args.n_agents)
            
            for i in range(args.n_agents):
                self.model_path = os.path.join(self.model_path, 'agent_%d' % i)
                self.actor_network[i].load_state_dict(torch.load(self.model_path + '/' + str(self.num) + '_actor_params.pkl'))


    def __call__(self, obs):

        actions = []
        probs = []
        act_env = []
        for i in range(self.args.n_agents):

            

        
            c_obs = torch.tensor(obs[i], dtype = torch.float)
            #print('the obs tensor', c_obs)

            act, _, probab = self.actor[i](c_obs)
            
            actions.extend(act)
            probs.append(probab)
            act_env.append(list(act))
       
        #print(actions)
        return actions, None, act_env


    def update_critic(self, sample, writer):
        self.args.train = True

        def unpack_batch(sample, agent_id):

            states, actions, rewards, terminals, next_states, global_states = [],[],[],[],[], []
            
            for _, exp in enumerate(sample):
                if type(exp) is tuple:
                    exp = exp[0]
                states.append(np.array((exp.state)))
                actions.append((exp.action))
                rewards.append(exp.reward)
                next_states.append(np.array((exp.state)))
                terminals.append(exp.terminal)

            return states, actions, \
                   np.array(rewards, dtype=np.float32), \
                   np.array(terminals, dtype=np.uint8), \
                   next_states

        states, actions, rewards, terminals, next_states = unpack_batch(sample, id)
        #print(states.shape)
        states = torch.FloatTensor(states).reshape(3, 2049, 18);  
        states = states.to(self.device)

        actions = torch.FloatTensor(actions); 
        actions= actions.to(self.device)

        next_states = torch.FloatTensor(next_states).reshape(3, 2049, 18);  
        next_states = next_states.to(self.device)

        #print(states.reshape(3, 2049, 18).shape)
        
        qs = self.critic(states)
        next_qs = self.target_critic(next_states)        
        q_loss = 0

        for a_i, values, main_value in zip(range(self.args.n_agents), next_qs,
                                                qs):

            
        
            #Calculate GAE(lambda) Advantage and TD(lambda) return
            # generalized advantage estimator: smoothed version of the advantage
            last_gae = 0.0
            adv_array = []
            value_array = []
            for val, next_val, done, reward in zip(reversed(values[:-1]),
                                                    reversed(values[1:]),
                                                        reversed(terminals[:-1]),
                                                        reversed(rewards[:-1])):
                if done:
                    delta = reward - val
                    last_gae = delta
                else:
                    delta = reward + self.args.gamma * next_val - val
                    last_gae = delta + self.args.gamma * self.args.gae_lambda * last_gae
                adv_array.append(last_gae)
                value_array.append(last_gae + val)
            advantages = torch.FloatTensor(list(reversed(adv_array))); advantages = advantages.to(self.device)
            td_values = torch.FloatTensor(list(reversed(value_array))); td_values = td_values.to(self.device)


            q_loss += F.mse_loss(main_value, td_values)

        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm(
            self.critic.parameters(), 10 * self.args.n_agents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        return advantages
        
    def update_actor(self, sample, advantages, writer):
        self.args.train = True
        def unpack_batch(sample, agent_id):

            states, actions, rewards, terminals, next_states, global_states = [],[],[],[],[], []
            
            for _, exp in enumerate(sample):
                if type(exp) is tuple:
                    exp = exp[0]
                states.append(np.array((exp.state)[agent_id]))
                actions.append((exp.action)[agent_id])
                rewards.append(exp.reward)
                next_states.append(np.array((exp.state)[agent_id]))
                terminals.append(exp.terminal)

            return states, actions, \
                   np.array(rewards, dtype=np.float32), \
                   np.array(terminals, dtype=np.uint8), \
                   next_states
    
        for id in range(self.args.n_agents):
            

            states, actions, rewards, terminals, next_states = unpack_batch(sample, id)
            states = torch.FloatTensor(states);  
            states = states.to(self.device)

            actions = torch.FloatTensor(actions); 
            actions= actions.to(self.device)
        
            #Calculate GAE(lambda) Advantage and TD(lambda) return
            # generalized advantage estimator: smoothed version of the advantage
            
            
            # normalize advantages
            advantages = advantages - torch.mean(advantages)
            advantages /= torch.std(advantages)
            
            #Calculate old_logprob
          
            old_logprob, _ = self.actor[id].evaluate_actions(states, actions)
            
            # drop last entry from the trajectory, an our adv and ref value calculated without it
            #trajectory = trajectory_buffer[:-1] #check it without while training
            old_logprob = old_logprob[:-1].detach()
            
            for epoch in range(self.args.ppo_epoches):
                for batch_ofs in range(0, len(sample)-1, self.args.ppo_batch_size):
                    batch_ofs_end = batch_ofs + self.args.ppo_batch_size
                    
                    state_batch = states[batch_ofs:batch_ofs_end]
                    actions_batch = actions[batch_ofs:batch_ofs_end]
                    advantages_batch = advantages[batch_ofs:batch_ofs_end]
                    advantages_batch = advantages_batch.unsqueeze(-1)
                    old_logprob_batch =  old_logprob[batch_ofs:batch_ofs_end]

                    # critic training
                    
                    
                    # actor training
                    self.actor_optimizer[id].zero_grad()
                    log_prob, entropy = self.actor[id].evaluate_actions(state_batch, actions_batch)
                    #ratio between old and new policy
                    ratio = torch.exp(log_prob - old_logprob_batch)
                    #Clipped surrogate loss
                    policy_loss_1 = advantages_batch * ratio
                    policy_loss_2 = advantages_batch * torch.clamp(ratio,1.0 - self.args.ppo_eps, 1.0 + self.args.ppo_eps)
                    actor_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                    actor_loss = actor_loss - entropy*self.args.entropy_coeff
                    actor_loss.backward()
                    self.actor_optimizer[id].step()

            writer.add_scalar('actor_loss_{}'.format(id), actor_loss, self.niter)
            
        


            

        


