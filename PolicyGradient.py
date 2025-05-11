from agent import Agent, Experience
import argparse
import collections
import gymnasium as gym
import logging
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.types as torch_types
from typing import List, Dict, Tuple, Any, Optional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def LunarLander_preprocess_experiences(experiences: List[Experience]) -> List[Experience]:
    # Preprocess the experience for LunarLander
    # Increase the reward for action 0 (noop) if the lander is in a good position
    total_reward = 0
    for i in range(len(experiences)):
        state, action, reward, new_state, done, log_prob = experiences[i]
        total_reward += reward
        if abs(state[0]) < 0.1 and abs(state[1]) < 0.01 and abs(state[3]) < 0.1 and total_reward > 150 and action == 0 and reward > 0:
            reward *= 2
        experiences[i] = Experience(state, action, reward, new_state, done, log_prob)
    return experiences

hyperparameters = {
    'default': {
        'layers': [],
        'gamma': 0.99,
        'lr': 0.01,
        'prep': None
    },
    'CartPole':  {
        'layers': [16],
        'gamma': 1.0,
        'lr': 0.01,
        'prep': None
    },
    'LunarLander':  {
        'layers': [8],
        'gamma': 0.99,
        'lr': 0.01,
        'prep': LunarLander_preprocess_experiences,
    },
}

class PolicyGradientAgent(nn.Module, Agent):
    def __init__(self, s_size: int, a_size: int, h_sizes: List[int], hp : Dict[str, Any]={}):
        super(PolicyGradientAgent, self).__init__()
        self._layers = nn.ModuleList([nn.Linear(*lu) for lu in zip([s_size] + h_sizes, h_sizes + [a_size])])
        self._optimizer = None
        self._hp = hp
        self._gamma = hp['gamma']
        logging.debug(f'layers: {self._layers}')

    def forward(self, x: torch.Tensor):
        for layer in self._layers[:-1]:
            x = F.relu(layer(x))
        x = self._layers[-1](x)
        return F.softmax(x, dim=1)

    def act(self, state: np.ndarray) -> Tuple[torch_types.Number, Optional[torch.Tensor]]:
        t_state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(t_state).cpu()
        m = dist.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def reinforce(self, experiences: collections.deque[Experience], new_experience: int):
        if not self._optimizer:
            self._optimizer = optim.Adam(self.parameters(), lr=self._hp['lr'])

        if self._hp['prep']:
            experiences = self._hp['prep'](experiences)
        
        n_step = len(experiences)
        returns = collections.deque(maxlen=n_step)
        rewards = [e.reward for e in experiences]
        log_probs = [e.extra for e in experiences]
        
        for t in reversed(range(n_step)):
            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft( self._gamma * disc_return_t + rewards[t] )
        
        eps = np.finfo(np.float32).eps.item()
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        policy_loss = []
        for (log_prob, disc_return) in zip(log_probs, returns):
            policy_loss.append(-log_prob * disc_return) # type: ignore
        policy_loss = torch.cat(policy_loss).sum()
        
        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()
        experiences.clear()
    
    def get_state_dict(self) -> Dict[str, Any]:
        state = {
            'model': self.state_dict(),
            'hp': self._hp.copy(),
        }
        del state['hp']['prep']
        if self._optimizer:
            state['optimizer'] = self._optimizer.state_dict()
        return state

def create_agent(env: gym.Env, args: List[str]) -> Agent:
    envId = env.spec.id # type: ignore
    s_size = env.observation_space.shape[0] # type: ignore
    a_size = env.action_space.n # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--layers', type=int, nargs='*', help='an integer for the accumulator')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--gamma', help='discount rate for reward')
    parsed_args = parser.parse_args(args)
    
    hp = hyperparameters.get(envId.split('-')[0], hyperparameters['default'])
    if parsed_args.layers is not None:
        hp['layers'] = parsed_args.layers
    if parsed_args.lr:
        hp['lr'] = parsed_args.lr
    if parsed_args.gamma:
        hp['gamma'] = parsed_args.gamma

    print(f"Creating PolicyGradient for {envId} with layers: {hp['layers']}, gamma: {hp['gamma']}, lr: {hp['lr']}")
    
    return PolicyGradientAgent(s_size, a_size, hp['layers'], hp=hp)

def load_agent(env: gym.Env, state: Dict[str, Any]) -> Agent:
    envId = env.spec.id # type: ignore
    s_size = env.observation_space.shape[0] # type: ignore
    a_size = env.action_space.n # type: ignore

    hp = hyperparameters.get(envId.split('-')[0], hyperparameters['default']) | state.get('hp', {})
    
    agent = PolicyGradientAgent(s_size, a_size, hp['layers'], hp=hp)
    agent.load_state_dict(state['model'])
    if 'optimizer' in state:
        agent._optimizer = optim.Adam(agent.parameters(), lr=hp['lr'])
        agent._optimizer.load_state_dict(state['optimizer'])
    
    print(f"Loading PolicyGradient for {envId} with layers: {hp['layers']}, gamma: {hp['gamma']}, lr: {hp['lr']}")

    return agent