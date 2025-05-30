from agent import Action, Agent, Experience
import argparse
import gymnasium as gym
import logging
import numpy as np
import PolicyGradient
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, MutableSequence, Sequence, Any

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyperparameters = PolicyGradient.hyperparameters
for key in hyperparameters:
    hyperparameters[key].update({
        'c_layers': [16],
        'beta': 0.01,
        'critic_anchor_strength': 0.0, # L2 regularization towards old critic weights
        'normalize_returns': False,
    })
hyperparameters['CartPole'].update({
    'c_layers': [16],
    'beta': 0.01,
    'critic_anchor_strength': 0.0,
    'normalize_returns': False,
})
hyperparameters['LunarLander'].update({
    'c_layers': [16],
    'beta': 0.001,
    'critic_anchor_strength': 0.001,
    'normalize_returns': False,
})
hyperparameters['Pixelcopter'].update({
    'lr': 0.001,
    'c_layers': [64],
    'beta': 0.001,
    'critic_anchor_strength': 0.001,
    'normalize_returns': False,
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
        self._critic_anchor_strength = self._hp.get('critic_anchor_strength', 0.0)
        self._critic_anchor_weights = None
        self._critic_losses = []

        self._critic_anchor_weights = None

    def act(self, state: torch.Tensor) -> Action:
        action, log_prob, value = super(ActorCriticAgent, self).act(state)
        t_state = state.unsqueeze(0).to(device)
        value = self._critic(t_state).squeeze()
        return Action(action, log_prob, value)
    
    def train(self, train: bool):
        super(ActorCriticAgent, self).train(train)
        self._critic.train(train)
        
    def values(self, experiences: Sequence[Sequence[Experience]]) -> torch.Tensor:
        values = torch.stack([torch.stack([exp.value if exp.value is not None else self._critic(exp.state.unsqueeze(0)).squeeze() for exp in traj ]) for traj in experiences]).to(device)
        return values
    
    def next_values(self, experiences: Sequence[Sequence[Experience]]) -> torch.Tensor:
        next_values = torch.stack([torch.stack([torch.tensor(0.0) if exp.done else (exp.next_value if exp.next_value is not None else self._critic(exp.next_state.unsqueeze(0)).squeeze().detach()) for exp in traj ]) for traj in experiences]).to(device)
        return next_values
    
    def update_critic_weights(self):
        if self._critic_anchor_strength > 0:
            self._critic_anchor_weights = [p.clone().detach().cpu() for p in self._critic.parameters()]
            logging.info(f"Critic anchor weights updated (strength: {self._critic_anchor_strength}).")
        else:
            self._critic_anchor_weights = None

    def reinforce_critic(self, values: torch.Tensor, targets: torch.Tensor):
        if self._critic_optimizer is None:
            self._critic_optimizer = optim.Adam(
                self._critic.parameters(),
                lr=self._hp['beta'],
                weight_decay=self._hp.get('critic_weight_decay', 0.0)
            )
        critic_loss = F.mse_loss(values, targets)
        logging.debug(f'Critic loss (before anchor): {critic_loss.item()}')

        total_critic_loss = critic_loss
        if self._critic_anchor_strength > 0 and self._critic_anchor_weights is not None:
            critic_anchor_reg_loss = 0.0
            for param, anchor_param in zip(self._critic.parameters(), self._critic_anchor_weights):
                critic_anchor_reg_loss += torch.sum((param - anchor_param.to(param.device))**2)
            logging.debug(f'Critic anchor regularization term: {critic_anchor_reg_loss.item()}')
            total_critic_loss += self._critic_anchor_strength * critic_anchor_reg_loss

        self._critic_losses.append(total_critic_loss.item())
        self._critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self._critic_optimizer.step()
        self.update_critic_weights()  # Update anchor weights after each training step
 
    def compute_advantage(self, returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        self.reinforce_critic(values, returns.detach())
        advantages = returns.detach() - values.detach()
        logging.debug(F'advantages:\n{advantages}')
        return advantages.detach()
    
    def reinforce(self, experiences: MutableSequence[Experience], new_experiences: int):
        logging.info(f"Monte Carlo reinforce: {len(experiences)} experiences, {new_experiences} new experiences")
        super(ActorCriticAgent, self).reinforce(experiences, new_experiences)
    
    def get_state_dict(self) -> Dict[str, Any]:
        state = super(ActorCriticAgent, self).get_state_dict()
        state['critic'] = {key: value.cpu() for key, value in self._critic.state_dict().items()}
        if self._critic_optimizer:
            state['critic_optimizer'] = self._critic_optimizer.state_dict()
        return state
    
    def load_state_dict(self, state: Dict[str, Any]):
        self._critic.load_state_dict(state['critic'])
        self.update_critic_weights()
        if 'critic_optimizer' in state:
            self._critic_optimizer = optim.Adam(
                self._critic.parameters(),
                lr=self._hp['beta']
            )
            self._critic_optimizer.load_state_dict(state['critic_optimizer'])
        else:
            self._critic_optimizer = None
        super(ActorCriticAgent, self).load_state_dict(state)

    def learning_metrics(self) -> str:
        result = super(ActorCriticAgent, self).learning_metrics()
        if self._critic_losses:
            result += f' critic_loss mean: {np.mean(self._critic_losses):.2f}; std: {np.std(self._critic_losses):.2f}'
            self._critic_losses.clear()
        return result

def create_agent(env: gym.Env, args: List[str]) -> Agent:
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--layers', type=int, nargs='*', help='number of unit for actor hidden layers')
    parser.add_argument('-c', '--c-layers', type=int, nargs='*', help='number of unit for critic hidden layers')
    parser.add_argument('--alpha', type=float, help='learning rate for actor')
    parser.add_argument('--beta', type=float, help='learning rate for critic')
    parser.add_argument('--gamma', type=float, help='discount rate for reward')
    parser.add_argument('--critic-anchor-strength', type=float, help='Strength for L2 regularization of critic towards its initial weights')
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
    if parsed_args.critic_anchor_strength is not None:
        hp['critic_anchor_strength'] = parsed_args.critic_anchor_strength

    print(f"Creating {__name__} for {envId} with actor layers: {hp['layers']}, critic layers: {hp['c_layers']}, gamma: {hp['gamma']}, alpha: {hp['lr']}, beta: {hp['beta']}, critic_anchor_strength: {hp.get('critic_anchor_strength', 0.0)}")
    
    return ActorCriticAgent(s_size, a_size, hp=hp)

def load_agent(env: gym.Env, state: Dict[str, Any]) -> Agent:
    envId = env.spec.id # type: ignore
    s_size = env.observation_space.shape[0] # type: ignore
    a_size = env.action_space.n # type: ignore

    hp = hyperparameters.get(envId.split('-')[0], hyperparameters['default']) | state.get('hp', {})
    
    agent = ActorCriticAgent(s_size, a_size, hp=hp)
    agent.load_state_dict(state)
    
    print(f"Loading {__name__} for {envId} with layers: {hp['layers']}, critic layers: {hp['c_layers']}, gamma: {agent._gamma}, alpha: {hp['lr']}, beta: {hp['beta']}, critic_anchor_strength: {hp.get('critic_anchor_strength', 0.0)}")

    return agent

if __name__ == "__main__":
    create_agent(None, ['-h'])