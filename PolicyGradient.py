from agent import Action, Agent, Experience
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
        ex = experiences[i]
        total_reward += ex.reward
        if abs(ex.state[0]) < 0.1 and abs(ex.state[1]) < 0.01 and abs(ex.state[3]) < 0.1 and total_reward > 150 and ex.action == 0 and ex.reward > 0:
            experiences[i] = Experience(ex.state, ex.action, ex.reward * 2, ex.done, ex.truncated, ex.next_state, ex.next_action, ex.log_prob, ex.value, ex.next_value)
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
    
    def act(self, state: torch.Tensor) -> Action:
        t_state = state.unsqueeze(0).to(device)
        probs = self._actor(t_state).cpu()
        m = dist.Categorical(probs)
        action = m.sample()
        return Action(action, m.log_prob(action), torch.tensor(0, dtype=torch.float32))
    
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
    
    def compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor, truncates: torch.Tensor, next_values: torch.Tensor) -> torch.Tensor:
        assert rewards.shape == dones.shape
        assert rewards.shape == truncates.shape
        assert rewards.shape == next_values.shape

        n_step = rewards.shape[1]
        returns = torch.empty([rewards.shape[0], 0], dtype=torch.float32, device=device)
        disc_returns = next_values[:, -1]
        for t in reversed(range(n_step)):
            disc_returns = (self._gamma * (1 - dones[:,t]) * ((1 - truncates[:, t]) * disc_returns + truncates[:, t] * next_values[:, t]) + rewards[:,t])
            returns = torch.cat((disc_returns.unsqueeze(1), returns), dim=1)
            
        logging.info(F'returns:\n{returns}')
        return returns

    def normalize_return(self, returns: torch.Tensor) -> torch.Tensor:
        self._mean, self._variance, self._n_sample = mean_variance([self._mean, returns.mean().item()], [self._variance, returns.var().item()], [self._n_sample, returns.numel()])
        
        mean = self._mean
        std = np.sqrt(self._variance)
        returns = (returns - mean) / (std + eps)
        logging.info(F'returns:\n{returns}')
        return returns
        
    def compute_advantage(self, returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        advantages = returns - 0.0
        logging.info(F'advantages:\n{advantages}, advantages.shape: {advantages.shape}')
        return advantages
    
    def reinforce_actor(self, experiences: Sequence[Sequence[Experience]]):
        if not self._actor_optimizer:
            self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=self._hp['lr'])

        # Unzip experiences into separate components using list comprehension
        rewards = torch.tensor([[exp.reward for exp in traj] for traj in experiences]).float().to(device)
        dones = torch.tensor([[exp.done for exp in traj] for traj in experiences]).long().to(device)
        truncates = torch.tensor([[exp.truncated for exp in traj] for traj in experiences]).long().to(device)
        log_probs = torch.stack([torch.stack([exp.log_prob for exp in traj if exp.log_prob is not None]) for traj in experiences]).squeeze(2).to(device)
        values = torch.stack([torch.stack([exp.value for exp in traj if exp.value is not None]) for traj in experiences]).to(device)
        next_values = torch.stack([torch.stack([exp.next_value for exp in traj if exp.next_value is not None]) for traj in experiences]).to(device)
        logging.info(F'log_probs:\n{log_probs}, log_probs.shape: {log_probs.shape}')
        logging.info(F'rewards:\n{rewards}, rewards.shape: {rewards.shape}')
        logging.info(F'dones:\n{dones}, dones.shape: {dones.shape}')
        logging.info(F'truncates:\n{truncates}, truncates.shape: {truncates.shape}')
        logging.info(F'values:\n{values}, values.shape: {values.shape}')
        logging.info(F'next_values:\n{next_values}, next_values.shape: {next_values.shape}')
        
        # Compute returns
        returns = self.compute_returns(rewards, dones, truncates, next_values)
        
        normlized_returns = self.normalize_return(returns)

        advantages = self.compute_advantage(normlized_returns, values)
        
        logging.info(F'log_probs:\n{log_probs}, log_probs.shape: {log_probs.shape}')
        
        policy_loss = (- log_probs * advantages).sum()
        logging.info(f'policy_loss: {policy_loss}')
        
        self._actor_optimizer.zero_grad()
        policy_loss.backward()
        self._actor_optimizer.step()

    
    def reinforce(self, experiences: MutableSequence[Experience], new_experience: int):
        if self._hp['prep']:
            experiences = self._hp['prep'](experiences)
        self.reinforce_actor([experiences])
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
    create_agent(None, ['-h']) # type: ignore