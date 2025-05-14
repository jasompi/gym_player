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
        'gamma': 0.995,
        'lr': 0.01,
        'prep': LunarLander_preprocess_experiences,
    },
}


class Actor(nn.Module):
    def __init__(self, s_size: int, a_size: int, h_sizes: List[int]):
        super(Actor, self).__init__()
        self._layers = nn.ModuleList([nn.Linear(*lu) for lu in zip([s_size] + h_sizes, h_sizes + [a_size])])
        logging.debug(f'layers: {self._layers}')
    
    def forward(self, x: torch.Tensor):
        for layer in self._layers[:-1]:
            x = F.relu(layer(x))
        x = self._layers[-1](x)
        return F.softmax(x, dim=1)

    
class PolicyGradientAgent(Agent):
    def __init__(self, s_size: int, a_size: int, h_sizes: List[int], hp : Dict[str, Any]={}):
        super(PolicyGradientAgent, self).__init__()
        self._actor = Actor(s_size, a_size, h_sizes).to(device)
        self._actor_optimizer = None
        self._hp = hp
        self._gamma = hp['gamma']
 
    def train(self, train: bool):
        return self._actor.train(train)
    
    def act(self, state: np.ndarray) -> Tuple[torch_types.Number, Optional[torch.Tensor]]:
        t_state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self._actor(t_state).cpu()
        m = dist.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def _compute_returns_vec(self, rewards: List[float]) -> torch.Tensor:
        n_step = len(rewards)
        gamma = torch.tensor(self._gamma, device=device)
        t_rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        mask = torch.ones((n_step, n_step), device=device).triu()
        power = torch.arange(n_step, device=device)
        return (torch.pow(gamma, mask * power - power.unsqueeze(1)) * mask * t_rewards).sum(dim=1)
    
    def _compute_returns(self, rewards) -> torch.Tensor:
        n_step = len(rewards)
        returns = collections.deque(maxlen=n_step)
        for t in reversed(range(n_step)):
            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft( self._gamma * disc_return_t + rewards[t] )
        t_returns = torch.tensor(returns, dtype=torch.float32, device=device)
        logging.info(F'returns:\n{t_returns}')
        return t_returns
        
    def _compute_advantage(self, returns: torch.Tensor) -> torch.Tensor:
        eps = np.finfo(np.float32).eps.item()
        advantages = (returns - returns.mean()) / (returns.std() + eps)
        logging.info(F'advantages:\n{advantages}')
        return advantages

    def reinforce(self, experiences: collections.deque[Experience], new_experience: int):
        if not self._actor_optimizer:
            self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=self._hp['lr'])

        if self._hp['prep']:
            experiences = self._hp['prep'](experiences)
        
        rewards = [e.reward for e in experiences]

        returns = self._compute_returns(rewards)

        advantages = self._compute_advantage(returns)
        
        log_probs = torch.stack([e.extra for e in experiences if torch.is_tensor(e.extra)]).squeeze(1).to(device)
        logging.info(F'log_probs:\n{log_probs}')
        
        policy_loss = (- log_probs * advantages).sum()
        logging.info(f'policy_loss: {policy_loss}')
        
        self._actor_optimizer.zero_grad()
        policy_loss.backward()
        self._actor_optimizer.step()
        experiences.clear()
    
    def get_state_dict(self) -> Dict[str, Any]:
        state = {
            'model': {key: value.cpu() for key, value in self._actor.state_dict().items()},
            'hp': self._hp.copy(),
        }
        del state['hp']['prep']
        if self._actor_optimizer:
            state['optimizer'] = self._actor_optimizer.state_dict()
        return state
    
    def load_state_dict(self, state: Dict[str, Any]):
        self._actor.load_state_dict(state['model'])
        self._actor.to(device)
        if 'optimizer' in state:
            self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=self._hp['lr'])
            self._actor_optimizer.load_state_dict(state['optimizer'])


def create_agent(env: gym.Env, args: List[str]) -> Agent:
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--layers', type=int, nargs='*', help='an integer for the accumulator')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--gamma', help='discount rate for reward')
    parsed_args = parser.parse_args(args)
    
    envId = env.spec.id # type: ignore
    s_size = env.observation_space.shape[0] # type: ignore
    a_size = env.action_space.n # type: ignore

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
    agent.load_state_dict(state)
    print(f"Loading PolicyGradient for {envId} with layers: {hp['layers']}, gamma: {hp['gamma']}, lr: {hp['lr']}")

    return agent

if __name__ == "__main__":
    create_agent(None, ['-h'])