# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, env, parameters):
        super(Policy, self).__init__()
        self.env = env

    def act(self, state):
        return self.env.action_space.sample(), None
    
    def reinforce(self, replay_buffer):
        # No reinforcement learning for random policy
        pass
    
    def get_state_dict(self):
        return {}

def create_policy(env, parameters, verbose=0):
    if verbose:
        print(f"Creating random policy for {env.spec.id}")
    return Policy(env, parameters)

def load_policy(env, state, verbose=0):
    if verbose:
        print(f"Loading random policy for {env.spec.id}")

    return Policy(env, state)
