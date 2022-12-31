import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from itertools import chain

from utils import make_matrix, init, get_clones, ACTLayer



class HCritic(nn.Module):

    def __init__(self, args, sa_sizes, log = None):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(HCritic, self).__init__()
        
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = args.attend_heads
        self.hidden_dim = args.hidden_dim
        self.norm_in = args.norm_in
        self.args = args
        self.logger = log

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in self.sa_sizes:
          
            idim = sdim
            odim = 1
            encoder = nn.Sequential()
            if self.norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, self.hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * self.hidden_dim,
                                                      self.hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(self.hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if self.norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            self.hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = self.hidden_dim // self.attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(self.attend_heads):
            self.key_extractors.append(nn.Linear(self.hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(self.hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(self.hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):

        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
       
        
        inps = [torch.tensor(s) for s in inps]
        states = [torch.tensor(s) for s in inps]

        print(states.shape)
      
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        mean_attend = []
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]

                #print('keys', torch.stack(keys).shape)
                #print('values', torch.stack(values).shape)

                if not self.args.train: 
                #selector = selector.reshape(selector.shape[0], -1)
                    selector = selector.reshape(1, selector.shape[0])
                    
                    #print(selector.shape)
                    
                    # calculate attention across agents
                    attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                                torch.stack(keys).unsqueeze(1).permute(1, 2, 0))
                else:
                    attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))

                #print(attend_logits.shape)

                #attend_logits = torch.matmul(selector,
                                             #torch.stack(keys).permute(1, 0))                             
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(2)
                #print('len of agent _{}:'.format(i), scaled_attend_logits.detach().numpy().shape)


                
                attend_weights = F.softmax(scaled_attend_logits, dim=2)

                #print(torch.stack(values).permute(0, 1).shape)

                if not self.args.train:

                    other_values = (torch.stack(values).unsqueeze(1).permute(1, 2, 0) *
                                    attend_weights).sum(dim=2)
                else:
                    #print('values', torch.stack(values).shape)
                    other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)

                #print(other_values.shape)
                
                #mean_attend.append(torch.mean(attend_weights, dim = 0))
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        #attention_matrix = torch.cat(mean_attend, dim = 0)
        #attention_matrix = make_matrix(self.args, attention_matrix).unsqueeze(0)
        #if self.logger is not None:
            #self.logger.add_image('Attended weights', attention_matrix, niter)
        all_rets = []
        for i, a_i in enumerate(agents):
            
            # calculate Q for each action
            
            if not self.args.train:
                critic_in = torch.cat((s_encodings[i].unsqueeze(0), *other_all_values[i]), dim=1)
            else:
                #print(s_encodings[i].shape)
                #print(torch.stack(other_all_values[i]).shape)
                critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)

            #print(critic_in.shape)
            all_q = self.critics[a_i](critic_in)


            all_rets.append(all_q)
        
        return all_rets

class AttentionCritic(nn.Module):

    def __init__(self, args, sa_sizes, log = None):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        assert (args.hidden_dim % args.attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = args.attend_heads
        self.hidden_dim = args.hidden_dim
        self.norm_in = args.norm_in
        self.args = args
        self.logger = log

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in self.sa_sizes:
          
            idim = sdim 
            odim = 1
            encoder = nn.Sequential()
            if self.norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, self.hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * self.hidden_dim,
                                                      self.hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(self.hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if self.norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            self.hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = self.hidden_dim // self.attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(self.attend_heads):
            self.key_extractors.append(nn.Linear(self.hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(self.hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(self.hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):

        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [torch.tensor(s) for s in inps]

        inps = [torch.tensor(s) for s in inps]

        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        mean_attend = []
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]

                #print(torch.stack(keys).shape)
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                #print('len of agent _{}:'.format(i), scaled_attend_logits.detach().numpy().shape)
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)

                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent

        all_rets = []
        for i, a_i in enumerate(agents):
            #print('attention_weights: ', np.array(all_attend_probs).shape)

            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            #print(critic_in.shape)
            all_q = self.critics[a_i](critic_in)
            all_rets.append(all_q)

        return all_rets

class Critc(nn.Module):

    def __init__(self, args, id):
        super(Critc, self).__init__()

        self.fc1 = nn.Linear(args.obs_shape[id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, args.action_shape[id])

    def forward(self, state):

        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value



class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x

class MLP(nn.Module):
    def __init__(self, args, obs_dim, agent_id, cat_self=True, attn_internal=False):
        super(MLP, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_dim

        

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x



class Actor_p(nn.Module):

    def __init__(self, args, agent_id):
        super(Actor_p, self).__init__()

        obs_dim = args.obs_shape[agent_id]

        self.base = MLP(args, obs_dim, agent_id)

        self.act = ACTLayer(args, agent_id)

    
    def forward(self, obs):

   
        #print(obs)
        features = self.base(obs)


        actions, action_log_probs = self.act(features, deterministic=False)
        action_probs = self.act.get_probs(features)
        


        return np.array(actions), action_log_probs.cpu(), np.array(action_probs.cpu())

    def evaluate_actions(self, obs, action):

        features = self.base(obs)


        action_log_probs, dist_entropy = self.act.evaluate_actions(features, action)

        return action_log_probs, dist_entropy


class Critic_p(nn.Module):

    def __init__(self, args, agent_id):
        super(Critic_p, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]

        if not args.global_state:
            obs_dim = args.obs_shape[agent_id]
        else:
            obs_dim = args.obs_global_shape[0]


        self.base = MLP(args, obs_dim, agent_id)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        
        self.v_out = init_(nn.Linear(args.hidden_size, 1))

        #args.action_shape[agent_id] - if for COMA
    
    def forward(self, obs):

        features = self.base(obs)

        return self.v_out(features)

