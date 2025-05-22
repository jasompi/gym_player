from agent import Action, Agent, Experience
import argparse
import gymnasium as gym
import logging
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, MutableSequence, Sequence, Any

eps = np.finfo(np.float32).eps.item()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mean_variance(means: Sequence[float], variances: Sequence[float], n_samples: Sequence[int]):
    n_sample = np.sum(n_samples)
    mean = np.sum(np.array(means) * np.array(n_samples)) / n_sample
    variance = np.sum((np.array(variances)  + np.square(np.array(means) - mean)) * np.array(n_samples)) / n_sample
    return mean, variance, n_sample

def LunarLander_preprocess_experiences(experiences: MutableSequence[Experience], new_experiences: int) -> MutableSequence[Experience]:
    # Preprocess the experience for LunarLander
    # Increase the reward for action 0 (noop) if the lander is in a good position
    total_reward = 0
    for i in range(len(experiences) - new_experiences, len(experiences)):
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
    'CartPole': {
        'layers': [16],
        'gamma': 1.0,
        'lr': 0.01,
        'prep': None
    },
    'LunarLander': {
        'layers': [8],
        'gamma': 0.99,
        'lr': 0.01,
        'prep': LunarLander_preprocess_experiences,
    },
    'Pixelcopter': {
        'layers': [64],
        'gamma': 0.99,
        'lr': 0.0001,
        'prep': None
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
        self._normalize_returns = hp.get('normalize_returns', True)
        self._actor_losses = []
        logging.info(f'normlize_returns: {self._normalize_returns}')

    def train(self, train: bool):
        return self._actor.train(train)
    
    def act(self, state: torch.Tensor) -> Action:
        t_state = state.unsqueeze(0).to(device)
        probs = self._actor(t_state).cpu()
        m = dist.Categorical(probs)
        action = m.sample()
        return Action(action, m.log_prob(action), None)
    
    def compute_returns_from_experiences(self, experiences: Sequence[Sequence[Experience]]) -> torch.Tensor:
        rewards = torch.tensor([[exp.reward for exp in traj] for traj in experiences]).float().to(device)
        dones = torch.tensor([[exp.done for exp in traj] for traj in experiences]).long().to(device)
        truncates = torch.tensor([[exp.truncated for exp in traj] for traj in experiences]).long().to(device)
        next_values = self.next_values(experiences)
 
        logging.debug(F'rewards:\n{rewards}, rewards.shape: {rewards.shape}')
        logging.debug(F'dones:\n{dones}, dones.shape: {dones.shape}')
        logging.debug(F'truncates:\n{truncates}, truncates.shape: {truncates.shape}')
        logging.debug(F'next_values:\n{next_values}, next_values.shape: {next_values.shape}')
        return self.compute_returns(rewards, dones, truncates, next_values)
        
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
            
        logging.debug(F'returns:\n{returns}')
        return returns

    def normalize_returns(self, returns: torch.Tensor) -> torch.Tensor:
        self._mean, self._variance, self._n_sample = mean_variance([self._mean, returns.mean().item()], [self._variance, returns.var().item()], [self._n_sample, returns.numel()])
        
        mean = self._mean
        std = np.sqrt(self._variance)
        returns = (returns - mean) / (std + eps)
        logging.debug(F'returns:\n{returns}')
        return returns
        
    def compute_advantage(self, returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        advantages = returns - 0.0
        logging.debug(F'advantages:\n{advantages}, advantages.shape: {advantages.shape}')
        return advantages
    
    def values(self, experiences: Sequence[Sequence[Experience]]) -> torch.Tensor:
        values = torch.stack([torch.stack([exp.value if exp.value is not None else torch.tensor(0) for exp in traj ]) for traj in experiences]).to(device)
        return values
    
    def next_values(self, experiences: Sequence[Sequence[Experience]]) -> torch.Tensor:
        next_values = torch.stack([torch.stack([exp.next_value if exp.next_value is not None else torch.tensor(0) for exp in traj ]) for traj in experiences]).to(device)
        return next_values
    
    def reinforce_actor(self, experiences: Sequence[Sequence[Experience]]):
        if not self._actor_optimizer:
            self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=self._hp['lr'])

        # Unzip experiences into separate components using list comprehension
        log_probs = torch.stack([torch.stack([exp.log_prob for exp in traj if exp.log_prob is not None]) for traj in experiences]).squeeze(2).to(device)
        logging.debug(F'log_probs:\n{log_probs}, log_probs.shape: {log_probs.shape}')

        values = self.values(experiences)
        logging.debug(F'values:\n{values}, values.shape: {values.shape}')
        
        # Compute returns
        returns = self.compute_returns_from_experiences(experiences)
        
        if self._normalize_returns:
            returns = self.normalize_returns(returns)

        advantages = self.compute_advantage(returns, values)
        
        logging.debug(F'log_probs:\n{log_probs}, log_probs.shape: {log_probs.shape}')
        
        policy_loss = (- log_probs * advantages).sum()
        self._actor_losses.append(policy_loss.item())
        logging.debug(f'policy_loss: {policy_loss}')
        
        self._actor_optimizer.zero_grad()
        policy_loss.backward()
        self._actor_optimizer.step()

    def prep_experiences(self, experiences: MutableSequence[Experience], new_experiences: int):
        if self._hp['prep']:
            experiences = self._hp['prep'](experiences, new_experiences)
        
    def reinforce(self, experiences: MutableSequence[Experience], new_experiences: int):
        self.prep_experiences(experiences, new_experiences)
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
            
    def learning_metrics(self) -> str:
        if self._actor_losses:
            result = f'actor loss mean: {np.mean(self._actor_losses):.4f} std: {np.std(self._actor_losses):.4f};'
            self._actor_losses.clear()
            return result
        return ""

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