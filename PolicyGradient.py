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
from typing import Dict, List, MutableSequence, Sequence, Tuple, Any, Optional

eps = np.finfo(np.float32).eps.item()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mean_variance(means: Sequence[float], variances: Sequence[float], n_samples: Sequence[int]):
    n_sample = np.sum(n_samples)
    mean = np.sum(np.array(means) * np.array(n_samples)) / n_sample
    variance = np.sum((np.array(variances)  + np.square(np.array(means) - mean)) * np.array(n_samples)) / n_sample
    return mean, variance, n_sample

def LunarLander_preprocess_experiences(experiences: MutableSequence[Experience]) -> MutableSequence[Experience]:
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
    def __init__(self, s_size: int, a_size: int, hp : Dict[str, Any]={}):
        super(PolicyGradientAgent, self).__init__()
        self._actor = Actor(s_size, a_size, hp['layers']).to(device)
        self._actor_optimizer = None
        self._hp = hp
        self._gamma = hp['gamma']
        self._mean = hp.get('mean', 0)
        self._variance = hp.get('variance', 0)
        self._n_sample = hp.get('nSample', 0)

    def train(self, train: bool):
        return self._actor.train(train)
    
    def act(self, state: np.ndarray) -> Tuple[torch_types.Number, Optional[torch.Tensor]]:
        t_state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self._actor(t_state).cpu()
        m = dist.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def _compute_returns_vec(self, experiences: Sequence[Experience]) -> torch.Tensor:
        rewards = [e.reward for e in experiences]
        n_step = len(rewards)
        gamma = torch.tensor(self._gamma, device=device)
        t_rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        mask = torch.ones((n_step, n_step), device=device).triu()
        power = torch.arange(n_step, device=device)
        t_returns = (torch.pow(gamma, mask * power - power.unsqueeze(1)) * mask * t_rewards).sum(dim=1)
        mean = self._mean
        std = np.sqrt(self._variance)
        t_returns = (t_returns - mean) / (std + eps)
        return t_returns
    
    def compute_returns(self, experiences: Sequence[Experience], next_return: float=0.0) -> torch.Tensor:
        rewards = [e.reward for e in experiences]
        dones = [e.done for e in experiences]
        n_step = len(rewards)
        returns = collections.deque(maxlen=n_step)
        disc_return = next_return
        for t in reversed(range(n_step)):
            disc_return = self._gamma * disc_return * (1 - dones[t]) + rewards[t]
            returns.appendleft( disc_return )
        t_returns = torch.tensor(returns, dtype=torch.float32, device=device)
        self._mean, self._variance, self._n_sample = mean_variance([self._mean, t_returns.mean().item()], [self._variance, t_returns.var().item()], [self._n_sample, n_step])
        
        mean = self._mean
        std = np.sqrt(self._variance)
        t_returns = (t_returns - mean) / (std + eps)
        logging.info(F'returns:\n{t_returns}')
        return t_returns
        
    def compute_advantage(self, returns: torch.Tensor, experiences: Sequence[Experience]) -> torch.Tensor:
        advantages = returns - 0
        logging.info(F'advantages:\n{advantages}')
        return advantages

    def next_return(self, next_state: np.ndarray, next_action: torch_types.Number|None = None) -> float:
        return 0.0
    
    def reinforce(self, experiences: MutableSequence[Experience], new_experience: int):
        if not self._actor_optimizer:
            self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=self._hp['lr'])

        if self._hp['prep']:
            experiences = self._hp['prep'](experiences)
        
        _, _, _, new_state, done, _ = experiences[-1]
        returns = self.compute_returns(experiences, 0 if done else self.next_return(new_state))

        advantages = self.compute_advantage(returns, experiences)
        
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
        state['hp'].update({
            'mean': self._mean,
            'variance': self._variance,
            'n_sample': self._n_sample
        })
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
    parser.add_argument('-l', '--layers', type=int, nargs='*', help='number of unit for hidden layers')
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
    logging.debug(f'hp: {hp}')
    print(f"Creating PolicyGradient for {envId} with layers: {hp['layers']}, gamma: {hp['gamma']}, lr: {hp['lr']}")
    
    return PolicyGradientAgent(s_size, a_size, hp=hp)

def load_agent(env: gym.Env, state: Dict[str, Any]) -> Agent:
    envId = env.spec.id # type: ignore
    s_size = env.observation_space.shape[0] # type: ignore
    a_size = env.action_space.n # type: ignore

    hp = hyperparameters.get(envId.split('-')[0], hyperparameters['default']) | state.get('hp', {})
    logging.debug(f'hp: {hp}')
    agent = PolicyGradientAgent(s_size, a_size, hp=hp)
    agent.load_state_dict(state)
    print(f"Loading PolicyGradient for {envId} with layers: {hp['layers']}, gamma: {hp['gamma']}, lr: {hp['lr']}")

    return agent

if __name__ == "__main__":
    create_agent(None, ['-h'])