from ast import Param
import gym
from matplotlib.pyplot import hist
import torch
import random
import collections
from torch.autograd import Variable
import torch as T
import threading


import numpy as np

from collections import namedtuple, deque


# one single experience step
# Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state','terminal', 'probs'])      


class MOARLExperienceSource:
    """
    Common Experience source both for MORL and MARL
    """
    def __init__(self, env, agents, args, steps_count=1, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param vectorized: support of vectorized envs from OpenAI universe
        """
        #assert isinstance(env, (gym.Env, list, tuple))
        #assert isinstance(agents, (BaseAgent, list[BaseAgent])) #changed the agents to list of agents
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        self.env = env
        self.agents = agents
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.vectorized = vectorized
        self.args = args
        self.train_profiles = None
        #self.buffer = buffer

    def __iter__(self):
        history, cur_rewards = [], 0
        iter_idx = 0
        if self.args.n_agents != 1:
            
            s = self.env.reset() 
            rewards_ind = [0]*self.args.n_agents
        else:
            s = self.env.reset() #state.
            if self.args.reward_size > 1:
                cur_rewards = np.zeros((len(self.env.reward_space))) 
            else:
                cur_rewards = 0

        
        
        while True:   
            u = []
            actions = []
             #------------------- Multi-Agent RL--------------------------
            if self.args.n_agents != 1:
               
 
                with torch.no_grad():

                    actions, probs, _ = self.agents(s)

                
                if self.args.one_hot:
                    values = actions
                    n_values = self.args.action_shape[0]
                    action = np.eye(n_values)[values]
                else:
                    action = actions
                
                

                s_next, r, dones, _ = self.env.step(action)

                #print(dones)

               

                cur_rewards += sum(r)

      
 
            

             
                history.append(Experience(state=s, action= actions, reward=r, next_state = s_next, terminal = dones, probs = probs))
                
                if len(history) == self.steps_count:
                    yield tuple(history)
                    
                    history.clear()
                s = s_next
                if not self.args.env_ending:
                    dones = [False]*self.args.n_agents
                if all(dones) or iter_idx % self.args.max_episode_len == 0:
                    #self.total_rewards = cur_rewards
              
                    
                    s = self.env.reset()
                    history.clear()
            
            
            else: #------------------- Multi-Objective RL--------------------------   
                #cur_rewards = np.zeros((len(self.env.reward_space)))
                with torch.no_grad():
                    action = self.agents(s, preference = None)
                    u = action
                s_next, r, done, info = self.env.step(u)
                cur_rewards += r
                if s is not None:
                    history.append(Experience(state=s, action=u, reward=r, next_state = s_next, terminal = done, global_state = [], probs = []))
                    
                if len(history) == self.steps_count:
                    yield tuple(history)
                
                    history.clear()
                
                s = s_next

                # if done or iter_idx % self.args.max_episode_len == 0:
                if done or iter_idx % self.args.max_episode_len == 0:
                    if self.train_profiles is not None:
                        load_profile_idx = random.choice(self.train_profiles)
                        s = self.env.reset(load_profile_idx = load_profile_idx)
                    else:
                        s = self.env.reset()
                        if self.args.reward_size > 1:
                            self.total_rewards.append(cur_rewards)
                            cur_rewards = np.zeros((len(self.env.reward_space))) 
                        else:
                            self.total_rewards = cur_rewards
                            cur_rewards = 0
                    history.clear()
            iter_idx += 1
            
    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
        return r

   
class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        #assert isinstance(experience_source, (MORLExperienceSource, MARLExperienceSource,  type(None)))
        assert isinstance(buffer_size, int)
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = deque()
        self.capacity = buffer_size

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size, random_sample = True):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        #return [self.buffer[key] for key in keys]
        if random_sample:
            return [self.buffer[key] for key in keys]
        else:
            return self.buffer

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.extend(sample)
        else:
            for _i in range(len(sample)): 
                self.buffer.popleft()
            self.buffer.extend(sample)

    def populate(self, samples):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)

    def clear(self):
        self.buffer = deque()