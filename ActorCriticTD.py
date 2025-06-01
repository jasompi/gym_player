import random
import ActorCriticMonteCarlo
from agent import Action, Agent, Experience
import argparse
import gymnasium as gym
import itertools
import logging
import torch
from typing import Dict, List, MutableSequence, Any

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MINI_BATCH_SIZE = 32
STEP_PER_UPDATE = 4

hyperparameters = ActorCriticMonteCarlo.hyperparameters
hyperparameters['default'].update({
    'beta': 0.01,
    'td_n': 1,
    'normalize_returns': False,
})
hyperparameters['CartPole'].update({
    'beta': 0.002,
    'td_n': 1,
    'normalize_returns': False,
})
hyperparameters['LunarLander'].update({
    'c_layers': [64, 64],
    'td_n': 1,
    'normalize_returns': False,
})
hyperparameters['Pixelcopter'].update({
    'c_layers': [64],
    'td_n': 1,
    'normalize_returns': False,
})
hyperparameters['Pong'].update({
    'c_layers': [16],
    'td_n': 1,
    'normalize_returns': False,
})

class ActorCriticTDAgent(ActorCriticMonteCarlo.ActorCriticAgent):
    def __init__(self, s_size: int, a_size: int, hp : Dict[str, Any]={}):
        super(ActorCriticTDAgent, self).__init__(s_size, a_size, hp)
        self._total_updates = hp.get('total_updates', 0)
        self._td_n = hp.get('td_n', 1)
        
    def act(self, state: torch.Tensor) -> Action:
        return super(ActorCriticMonteCarlo.ActorCriticAgent, self).act(state)
    
    def compute_advantage(self, returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        advantages = returns.detach() - values.detach()
        logging.debug(F'advantages:\n{advantages}')
        return advantages.detach()
        
    def reinforce(self, experiences: MutableSequence[Experience], new_experiences: int):
        logging.info(f"reinforce: {len(experiences)} experiences, {new_experiences} new experiences")
        self.prep_experiences(experiences, new_experiences)
        if new_experiences < self._td_n:
            return
        sample = range(len(experiences) - new_experiences, len(experiences) - self._td_n + 1)
        mini_batch = [[experiences[k] for k in range(i, i + self._td_n)] for i in sample]
        self.reinforce_actor(mini_batch)
        if len(experiences) < MINI_BATCH_SIZE:
            return
        for i in range(int(new_experiences / STEP_PER_UPDATE)):
            sample = random.sample(range(len(experiences)), MINI_BATCH_SIZE)
            mini_batch = [[experiences[k]] for k in sample]
            
            values = self.values(mini_batch)
            logging.debug(F'values:\n{values}, values.shape: {values.shape}')
            
            # Compute returns
            returns = self.compute_returns_from_experiences(mini_batch)
            self.reinforce_critic(values, returns.detach())
            self._total_updates += 1

    def get_state_dict(self) -> Dict[str, Any]:
        state = super(ActorCriticTDAgent, self).get_state_dict()
        state['critic'] = {key: value.cpu() for key, value in self._critic.state_dict().items()}
        state['total_updates'] = self._total_updates
        if self._critic_optimizer:
            state['critic_optimizer'] = self._critic_optimizer.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        self._total_updates = state['total_updates']
        super(ActorCriticTDAgent, self).load_state_dict(state)

def create_agent(env: gym.Env, args: List[str]) -> Agent:
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--layers', type=int, nargs='*', help='number of unit for actor hidden layers')
    parser.add_argument('-c', '--c-layers', type=int, nargs='*', help='number of unit for critic hidden layers')
    parser.add_argument('--alpha', type=float, help='learning rate for actor')
    parser.add_argument('--beta', type=float, help='learning rate for critic')
    parser.add_argument('--gamma', type=float, help='discount rate for reward')
    parser.add_argument('--n-td', type=int, help='number of temporal difference TD(n)')
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
    if parsed_args.n_td:
        hp['td_n'] = max(1, parsed_args.n_td)
    if len(__name__) > len('ActorCriticTD'):
        try:
            hp['td_n'] = max(1, int(__name__[len('ActorCriticTD'):]))
        except ValueError:
            logging.warning(f"Invalid td_n value from __name__: {__name__}. Using default value.")
    else:
        hp['td_n'] = 1

    print(f"Creating {__name__} for {envId} with actor layers: {hp['layers']}, critic layers: {hp['c_layers']}, gamma: {hp['gamma']}, alpha: {hp['lr']}, beta: {hp['beta']}, critic_anchor_strength: {hp.get('critic_anchor_strength', 0.0)}, td_n: {hp['td_n']}")
    
    return ActorCriticTDAgent(s_size, a_size, hp=hp)

def load_agent(env: gym.Env, state: Dict[str, Any]) -> Agent:
    envId = env.spec.id # type: ignore
    s_size = env.observation_space.shape[0] # type: ignore
    a_size = env.action_space.n # type: ignore

    hp = hyperparameters.get(envId.split('-')[0], hyperparameters['default']) | state.get('hp', {})
    
    agent = ActorCriticTDAgent(s_size, a_size, hp=hp)
    agent.load_state_dict(state)
    
    print(f"Loading {__name__} for {envId} with layers: {hp['layers']}, critic layers: {hp['c_layers']}, gamma: {agent._gamma}, alpha: {hp['lr']}, beta: {hp['beta']}, critic_anchor_strength: {hp.get('critic_anchor_strength', 0.0)}, total_updates: {agent._total_updates}, td_n: {agent._td_n}") 

    return agent

if __name__ == "__main__":
    create_agent(None, ['-h']) # type: ignore