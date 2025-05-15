from agent import Agent, Experience
import argparse
import collections
import gymnasium as gym
import logging
import numpy as np
import PolicyGradient
import random
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.types as torch_types
from typing import Dict, List, MutableSequence, Sequence, Tuple, Any, Optional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyperparameters = PolicyGradient.hyperparameters
hyperparameters['default'].update({
    'c_layers': [],
    'beta': 0.001,
})
hyperparameters['CartPole'].update({
    'c_layers': [16],
    'beta': 0.001,
})
hyperparameters['LunarLander'].update({
    'c_layers': [8],
    'beta': 0.001,
})

class Critic(nn.Module):
    def __init__(self, s_size: int, a_size: int, h_sizes: List[int]):
        super(Critic, self).__init__()
        self._layers = nn.ModuleList([nn.Linear(*lu) for lu in zip([s_size] + h_sizes, h_sizes + [a_size])])
        logging.debug(f'layers: {self._layers}')
        
    def forward(self, x: torch.Tensor):
        for layer in self._layers[:-1]:
            x = F.relu(layer(x))
        x = self._layers[-1](x)
        return x

class ActorCriticAgent(PolicyGradient.PolicyGradientAgent):
    def __init__(self, s_size: int, a_size: int, hp : Dict[str, Any]={}):
        super(ActorCriticAgent, self).__init__(s_size, a_size, hp)
        self._critic = Critic(s_size, 1, hp['c_layers']).to(device)
        self._critic_optimizer = None
        self._total_updates = hp.get('total_updates', 0)

    def train(self, train: bool):
        super(ActorCriticAgent, self).train(train)
        self._critic.train(train)
        
    def next_return(self, next_state: np.ndarray, next_action: torch_types.Number|None = None) -> float:
        # if next_action is None:
        #     next_action, _ = self.act(next_state)
        t_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        # next_return = self._critic(t_state).squeeze(1).gather(1, torch.tensor([next_action]).unsqueeze(1)).squeeze(1).item()
        next_return = self._critic(t_state).squeeze(1).item()
        logging.info(F'next_return: {next_return}')
        return next_return
 
    def compute_advantage(self, returns: torch.Tensor, experiences: Sequence[Experience]) -> torch.Tensor:
        if self._critic_optimizer is None:
            self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=self._hp['beta'])
            
        states = torch.from_numpy(np.array([e.state for e in experiences])).float().to(device)
        # actions = torch.from_numpy(np.array([e.action for e in experiences])).long().to(device)
        
        # values = self._critic(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        values = self._critic(states).squeeze(1)
        
        critic_loss = F.mse_loss(returns.detach(), values)
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        advantages = returns.detach() - values.detach()
        logging.info(F'advantages:\n{advantages}')
        return advantages.detach()
    
    def reinforce(self, experiences: MutableSequence[Experience], new_experiences: int):
        super(ActorCriticAgent, self).reinforce(experiences, new_experiences)
        
        self._total_updates += 1
        experiences.clear()
    
    def get_state_dict(self) -> Dict[str, Any]:
        state = super(ActorCriticAgent, self).get_state_dict()
        state['critic'] = {key: value.cpu() for key, value in self._critic.state_dict().items()}
        state['total_updates'] = self._total_updates
        if self._critic_optimizer:
            state['critic_optimizer'] = self._critic_optimizer.state_dict()
        return state
    
    def load_state_dict(self, state: Dict[str, Any]):
        self._critic.load_state_dict(state['critic'])
        self._total_updates = state['total_updates']
        if 'critic_optimizer' in state:
            self._critic_optimizer = optim.Adam(self._critic.parameters(), lr=self._hp['beta'])
            self._critic_optimizer.load_state_dict(state['critic_optimizer'])
        else:
            self._critic_optimizer = None
        super(ActorCriticAgent, self).load_state_dict(state)


def create_agent(env: gym.Env, args: List[str]) -> Agent:
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--layers', type=int, nargs='*', help='number of unit for actor hidden layers')
    parser.add_argument('-c', '--c-layers', type=int, nargs='*', help='number of unit for critic hidden layers')
    parser.add_argument('--alpha', type=float, help='learning rate for actor')
    parser.add_argument('--beta', type=float, help='learning rate for critic')
    parser.add_argument('--gamma', type=float, help='discount rate for reward')
    parsed_args = parser.parse_args(args)
    
    envId = env.spec.id # type: ignore
    s_size = env.observation_space.shape[0] # type: ignore
    a_size = env.action_space.n # type: ignore

    hp = hyperparameters.get(envId.split('-')[0], hyperparameters['default'])
    if parsed_args.layers is not None:
        hp['layers'] = parsed_args.layers or []
    if parsed_args.c_layers is not None:
        hp['c_layers'] = parsed_args.c_layers or []
    if parsed_args.alpha:
        hp['lr'] = parsed_args.alpha
    if parsed_args.beta:
        hp['beta'] = parsed_args.beta
    if parsed_args.gamma:
        hp['gamma'] = min(max(0.0, parsed_args.gamma), 1.0)

    print(f"Creating {__name__} for {envId} with actor layers: {hp['layers']}, critic layers: {hp['c_layers']}, gamma: {hp['gamma']}, alpha: {hp['lr']}, beta: {hp['beta']}")
    
    return ActorCriticAgent(s_size, a_size, hp=hp)

def load_agent(env: gym.Env, state: Dict[str, Any]) -> Agent:
    envId = env.spec.id # type: ignore
    s_size = env.observation_space.shape[0] # type: ignore
    a_size = env.action_space.n # type: ignore

    hp = hyperparameters.get(envId.split('-')[0], hyperparameters['default']) | state.get('hp', {})
    
    agent = ActorCriticAgent(s_size, a_size, hp=hp)
    agent.load_state_dict(state)
    
    print(f"Loading {__name__} for {envId} with layers: {hp['layers']}, critic layers: {hp['c_layers']}, gamma: {agent._gamma}, alpha: {hp['lr']}, beta: {hp['beta']}, total_updates: {agent._total_updates}")

    return agent

if __name__ == "__main__":
    create_agent(None, ['-h'])