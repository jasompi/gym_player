import argparse
import collections
import datetime

import gymnasium as gym
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyperparameters = {
    'default': {
        'layers': [],
        'gamma': 0.99,
        'lr': 0.01
    },
    'CartPole':  {
        "layers": [16],
        "gamma": 1.0,
        "lr": 0.01,
    },
}

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_sizes, hp={}):
        super(Policy, self).__init__()
        self._layers = nn.ModuleList([nn.Linear(*lu) for lu in zip([s_size] + h_sizes, h_sizes + [a_size])])
        self._optimizer = None
        # print(self._layers)
        self._hp = hp
        self._gamma = hp['gamma']

    def forward(self, x):
        for layer in self._layers[:-1]:
            x = F.relu(layer(x))
        x = self._layers[-1](x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = dist.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def reinforce(self, replay_buffer):
        if not self._optimizer:
            self._optimizer = optim.Adam(self.parameters(), lr=self._hp['lr'])

        n_step = len(replay_buffer)
        returns = collections.deque(maxlen=n_step)
        rewards = [t[2] for t in replay_buffer]
        log_probs = [t[4] for t in replay_buffer]
        
        for t in reversed(range(n_step)):
            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft( self._gamma * disc_return_t + rewards[t] )
        
        eps = np.finfo(np.float32).eps.item()
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        policy_loss = []
        for (log_prob, disc_return) in zip(log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()
        
        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()
    
    def get_state_dict(self):
        state = {
            'model': self.state_dict(),
            'hp': self._hp,
        }
        if self._optimizer:
            state['optimizer'] = self._optimizer.state_dict()
        return state

def create_policy(env, args, verbose=0):
    envId = env.spec.id
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--layers', type=int, nargs='*', help='an integer for the accumulator')
    parser.add_argument('-g', '--gamma', help='discount rate for reward')
    parser.add_argument('-L', '--lr', type=float, help='learning rate')
    args = parser.parse_args(args)
    hp = hyperparameters[envId.split('-')[0]] or hyperparameters['default']
    if args.layers is not None:
        hp['layers'] = args.layers
    if args.lr:
        hp['lr'] = args.lr
    if args.gamma:
        hp['gamma'] = args.gamma

    if verbose:
        print(f"Creating policy for {envId} with layers: {hp['layers']}, gamma: {hp['gamma']}, lr: {hp['lr']}")
    
    return Policy(s_size, a_size, hp['layers'], hp=hp)

def load_policy(env, state, verbose=0):
    envId = env.spec.id
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    hp = state['hp'] or hyperparameters[envId.split('-')[0]] or hyperparameters['default']
    
    policy = Policy(s_size, a_size, hp['layers'], hp=hp)
    policy.load_state_dict(state['model'])
    if 'optimizer' in state:
        policy._optimizer = optim.Adam(policy.parameters(), lr=hp['lr'])
        policy._optimizer.load_state_dict(state['optimizer'])
    
    if verbose:
        print(f"Loading policy for {envId} with layers: {hp['layers']}, gamma: {hp['gamma']}, lr: {hp['lr']}")

    return policy