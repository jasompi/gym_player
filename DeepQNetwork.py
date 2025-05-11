from agent import Agent, Experience
import argparse
import collections
import gymnasium as gym
import logging
import numpy as np
import random
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.types as torch_types
from typing import List, Dict, Tuple, Any, Optional

MINI_BATCH_SIZE = 32
STEP_PER_UPDATE = 4
UPDATES_PER_EPSILON_DECAY = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyperparameters = {
    'default': {
        'layers': [],
        'gamma': 0.99,
        'lr': 0.001,
        'tau': 0.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.95,
     },
    'CartPole':  {
        'layers': [16],
        'gamma': 1.0,
        'lr': 0.01,
        'tau': 0.001,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.95,
    },
    'LunarLander':  {
        'layers': [64, 64],
        'gamma': 0.995,
        'lr': 0.001,
        'tau': 0.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
    },
}

class DeepQNetwork(nn.Module):
    def __init__(self, s_size: int, a_size: int, h_sizes: List[int]):
        super(DeepQNetwork, self).__init__()
        self._layers = nn.ModuleList([nn.Linear(*lu) for lu in zip([s_size] + h_sizes, h_sizes + [a_size])])
        logging.debug(f'layers: {self._layers}')
        
    def forward(self, x: torch.Tensor):
        for layer in self._layers[:-1]:
            x = F.relu(layer(x))
        x = self._layers[-1](x)
        return x


class DQNAgent(Agent):
    def __init__(self, s_size: int, a_size: int, h_sizes: List[int], hp : Dict[str, Any]={}):
        super().__init__()
        self._q_network = DeepQNetwork(s_size, a_size, h_sizes).to(device)
        self._q_network.train(False)
        self._target_q_network = DeepQNetwork(s_size, a_size, h_sizes)
        self._target_q_network.load_state_dict(self._q_network.state_dict())
        self._target_q_network.to(device)
        self._target_q_network.train(False)
        self._hp = hp
        self._gamma = hp.get('gamma', 0.99)
        self._tau = hp.get('tau', 0.001)
        self._epsilon_min = hp.get('epsilon_min', 0.01)
        self._epsilon_decay = hp.get('epsilon_decay', 0.95)
        self._epsilon = hp.get('epsilon', 1.0)
        self._total_updates = hp.get('total_updates', 0)
        self._a_size = a_size
        self._optimizer = None
        self._train = False

    def train(self, train: bool):
        self._train = train
 
    def act(self, state: np.ndarray) -> Tuple[torch_types.Number, Optional[torch.Tensor]]:
        if self._train and np.random.rand() <= self._epsilon:
            return np.random.choice(self._a_size), None
        t_state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q_values = self._q_network.forward(t_state).cpu().detach().numpy()
        return np.argmax(q_values).item(), None

    def _update_target_network(self, tau: float):
        """Soft update of the target network parameters. If tau == 0, the soft update is not used.
        Instead a hard updated is performed when the ."""
        for target_param, param in zip(self._target_q_network.parameters(), self._q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
 
    def reinforce(self, experiences: List[Experience], new_experiences: int):
        if not self._optimizer:
            self._optimizer = optim.Adam(self._q_network.parameters(), lr=self._hp['lr'])
        if len(experiences) > MINI_BATCH_SIZE:
            self._q_network.train(True)
            for _ in range(int(new_experiences / STEP_PER_UPDATE)):
                mini_batch = random.sample(experiences, MINI_BATCH_SIZE)
                states = torch.from_numpy(np.array([e.state for e in mini_batch])).float().to(device)
                actions = torch.from_numpy(np.array([e.action for e in mini_batch])).long().to(device)
                rewards = torch.from_numpy(np.array([e.reward for e in mini_batch])).float().to(device)
                new_states = torch.from_numpy(np.array([e.new_state for e in mini_batch])).float().to(device)
                dones = torch.from_numpy(np.array([e.done for e in mini_batch])).float().to(device)

                q_values = self._q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = self._target_q_network(new_states).max(1)[0]
                expected_q_values = rewards + (self._gamma * next_q_values * (1 - dones))

                loss = F.mse_loss(q_values, expected_q_values.detach())
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                if self._tau > 0:
                    self._update_target_network(self._tau)
                self._total_updates += 1
                if self._total_updates % UPDATES_PER_EPSILON_DECAY == 0:
                    if self._tau == 0:
                        self._target_q_network.load_state_dict(self._q_network.state_dict())
                    self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
                    logging.info(f"Updated target network and epsilon: {self._epsilon}")
            self._q_network.train(False)
    
    def get_state_dict(self) -> Dict[str, Any]:
        state = {
            'model': {key: value.cpu() for key, value in self._q_network.state_dict().items()},
            'hp': self._hp.copy(),
        }
        state['hp']['epsilon'] = self._epsilon
        state['hp']['total_updates'] = self._total_updates
        if self._optimizer:
            state['optimizer'] = self._optimizer.state_dict()
        return state
    
    def load_state_dict(self, state: Dict[str, Any]):
        self._q_network.load_state_dict(state['model'])
        self._q_network = self._q_network.to(device)
        if 'optimizer' in state:
            self._optimizer = optim.Adam(self._q_network.parameters(), lr=self._hp['lr'])
            self._optimizer.load_state_dict(state['optimizer'])
        else:
            self._optimizer = None
        logging.debug(f"Loaded DQN agent with hyperparameters: {self._hp}")


def create_agent(env: gym.Env, args: List[str]) -> Agent:
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--layers', type=int, nargs='*', help='an integer for the accumulator')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--gamma', type=float, help='discount rate for reward')
    parser.add_argument('--tau', type=float, help='soft update rate. Use 0 for hard update')
    parser.add_argument('--epsilon-decay', type=float, help='epsilon decay rate')
    parsed_args = parser.parse_args(args)
    
    envId = env.spec.id # type: ignore
    s_size = env.observation_space.shape[0] # type: ignore
    a_size = env.action_space.n # type: ignore

    hp = hyperparameters.get(envId.split('-')[0], hyperparameters['default'])
    if parsed_args.layers is not None:
        hp['layers'] = parsed_args.layers or []
    if parsed_args.lr:
        hp['lr'] = parsed_args.lr
    if parsed_args.gamma:
        hp['gamma'] = min(max(0.0, parsed_args.gamma), 1.0)
    if parsed_args.tau is not None:
        hp['tau'] = min(max(0.0, parsed_args.tau), 1.0)
    if parsed_args.epsilon_decay:
        hp['epsilon_decay'] = min(max(0.0, parsed_args.epsilon_decay), 1.0)

    print(f"Creating DeepQNetwork for {envId} with layers: {hp['layers']}, gamma: {hp['gamma']}, lr: {hp['lr']}")
    
    return DQNAgent(s_size, a_size, hp['layers'], hp=hp)

def load_agent(env: gym.Env, state: Dict[str, Any]) -> Agent:
    envId = env.spec.id # type: ignore
    s_size = env.observation_space.shape[0] # type: ignore
    a_size = env.action_space.n # type: ignore

    hp = hyperparameters.get(envId.split('-')[0], hyperparameters['default']) | state.get('hp', {})
    
    agent = DQNAgent(s_size, a_size, hp['layers'], hp=hp)
    agent.load_state_dict(state)
    
    print(f"Loading DeepQNetwork for {envId} with layers: {hp['layers']}, gamma: {agent._gamma}, lr: {hp['lr']}, tau: {agent._tau}, epsilon_decay: {agent._epsilon_decay}, epsilon: {agent._epsilon}, total_updates: {agent._total_updates}")

    return agent

if __name__ == "__main__":
    create_agent(None, ['-h'])