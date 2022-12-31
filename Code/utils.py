import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import threading
import copy

def make_matrix(args, weights):
    n = args.n_agents
    t = torch.zeros(n, n)
    for i in range(n):
        k = 0
        for j in range(n):
            if i != j:
                t[i][j] = weights[i][k]
                k +=1
    return t


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def unpack_batch(batch_, args):


    lock = threading.Lock()
    memory = dict()
    for i in range(args.n_agents):
        memory['s_%d' % i] = np.empty([args.batch_size, args.obs_shape[i]])
        memory['a_%d' % i] = np.empty([args.batch_size])
        memory['r_%d' % i] = np.empty([args.batch_size])
        memory['s_next_%d' % i] = np.empty([args.batch_size, args.obs_shape[i]])
        memory['done_%d' % i] = np.empty([args.batch_size])


    for idx, exp in enumerate(batch_):

        if type(exp) is tuple:
            exp = exp[0]
        for i in range(args.n_agents):
            with lock:
                memory['s_%d' % i][idx] = (exp.state)[i]
                memory['a_%d' % i][idx] = (exp.action)[i]
                memory['r_%d' % i][idx] = (exp.reward)[i]
                memory['s_next_%d' % i][idx] = (exp.next_state)[i]
                memory['done_%d' % i][idx] = (exp.terminal)[i]
    return memory


def history(args, transitions):
     
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32) #batch transitions


        o, u, o_next, done, re, Gs, Gs_next = [], [], [], [], [], [], []

        for agent_id in range(args.n_agents):
            o.append(transitions['s_%d' % agent_id])
            u.append(transitions['a_%d' % agent_id])
            re.append(transitions['r_%d' % agent_id])
            o_next.append(transitions['s_next_%d' % agent_id])
            done.append(transitions['done_%d' % agent_id])
        


        return o, u, re, o_next, done

def history_ind(args, transitions, agent_id):
     
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32) #batch transitions


        o, u, o_next, done, re, Gs, Gs_next = [], [], [], [], [], [], []

        #for agent_id in range(args.n_agents):
        o = transitions['s_%d' % agent_id]
        u = transitions['a_%d' % agent_id]
        re = transitions['r_%d' % agent_id]
        o_next = transitions['s_next_%d' % agent_id]
        done = transitions['done_%d' % agent_id]
        


        return o, u, re, o_next, done

def test_env(args, env, agent, time_step, writer):
    state = env.reset()

    episode_reward = 0

    while True:
        args.train = False
        #env.render()
        with torch.no_grad():
            action, _, _ = agent(state)

        if args.one_hot:
            values = action
            n_values = args.action_shape[0]
            action = np.eye(n_values)[values]
        else:
            action = action
        state, reward, done, _= env.step(action)

        episode_reward += sum(reward)
        

        if all(done):
            #writer.add_scalar('test/episode_reward', episode_reward, time_step)
            print('episode_rewrad', episode_reward)
            break
        
    


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias






class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, args, agent_id):
        super(ACTLayer, self).__init__()
    
        self.multi_discrete = args.multi_discrete

        action_dim = args.action_shape[agent_id]

        if args.action_space[agent_id] == "Discrete":
            #action_dim = action_space.n
            self.action_out = Categorical(args.hidden_dim, action_dim, args.use_orthogonal, args.gain)
        elif args.action_space[agent_id] == "Box":
            #action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(args.hidden_dim, action_dim, args.use_orthogonal, args.gain)
        elif args.action_space[agent_id] == "MultiBinary":
            #action_dim = action_space.shape[0]
            self.action_out = Bernoulli(args.hidden_dim, action_dim, args.use_orthogonal, args.gain)
        elif args.action_space[agent_id] == "MultiDiscrete":
            self.multi_discrete = True
            #action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for dim in action_dim:
                self.action_outs.append(Categorical(args.hidden_dim, dim, args.use_orthogonal, args.gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        

    
    def forward(self, x, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """

        if self.multi_discrete:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)
        
        else:
            action_logits = self.action_out(x)
            actions = action_logits.mode() if deterministic else action_logits.sample() 
            action_log_probs = action_logits.log_probs(actions)
        
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :return action_probs: (torch.Tensor)
        """
        if self.multi_discrete:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """


        if self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                #print('the actions is', act)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1) # ! could be wrong
            dist_entropy = torch.tensor(dist_entropy).mean()
        
        else:
            action_logits = self.action_out(x, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy



def test_env_mpe(args, env, agents, writer, time_step):

    reward_test = 0
    
    
    Gs, s = env.reset()
    
    


    episode_reward = 0

    t = 0
    while True:
        with torch.no_grad():
            actions, _, _= agents(s)
        


        if args.one_hot:
            values = actions
            n_values = args.action_shape[0]
            action = np.eye(n_values)[values]
        else:
            action = actions
        
        (Gs_next, s_next), reward, done_n, _, _ = env.step(action)
    


        episode_reward += reward

        s = s_next

        #writer.add_scalar('average return', episode_reward/4, episode)
        t += 1
        if t > args.evaluate_episode_len:
            done_n = True
        
        if done_n:
            #print(episode_reward)

            writer.add_scalar('average_return_across_test_profiles', episode_reward, time_step) 

            
            break