import numpy as np
import torch
import random
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class ActionSelector:
    """
    Abstract class which converts scores to the actions
    """
    def __call__(self, scores):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        if len(scores.shape) == 1:
            return np.argmax(scores, axis=0)
        else:
            return np.argmax(scores, axis=1)

class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05, selector=None):
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    def __call__(self, scores):
        scores = scores.cpu().numpy()
        assert isinstance(scores, np.ndarray)
        if len(scores.shape) == 1:
            n_actions = scores.shape
            actions = self.selector(scores)
            if np.random.random() < self.epsilon:
                rand_actions = np.random.choice(n_actions[0])
                actions = rand_actions
            # actions = list(map(int, str(actions)))
        else:
            batch_size, n_actions = scores.shape
            actions = self.selector(scores)
            mask = np.random.random(size=batch_size) < self.epsilon
            rand_actions = np.random.choice(n_actions, sum(mask))
            actions[mask] = rand_actions
        return actions
    

class ProbabilityActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)

class GuassianNoiseActionSelector(ActionSelector):

    def __init__(self, args):
        self.args = args        

    def __call__(self, probs):
        
        noise = self.args.noise_rate * self.args.high_action * np.random.randn(*probs.shape)  # gaussian noise
        probs += noise
        u = np.clip(probs, -self.args.high_action, self.args.high_action)
        return u