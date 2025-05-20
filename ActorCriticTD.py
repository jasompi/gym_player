import ActorCriticMonteCarlo
from agent import Action, Agent, Experience
import argparse
import gymnasium as gym
import itertools
import logging
import torch
from typing import Dict, List, MutableSequence, Any

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyperparameters = ActorCriticMonteCarlo.hyperparameters
hyperparameters['default'].update({
    'beta': 0.01,
    'td_n': 1,
})
hyperparameters['CartPole'].update({
    'beta': 0.01,
    'td_n': 1,
})
hyperparameters['LunarLander'].update({
    'beta': 0.01,
    'td_n': 1,
})

class ActorCriticTDAgent(ActorCriticMonteCarlo.ActorCriticAgent):
    def __init__(self, s_size: int, a_size: int, hp : Dict[str, Any]={}):
        super(ActorCriticTDAgent, self).__init__(s_size, a_size, hp)
        self._total_updates = hp.get('total_updates', 0)
        self._td_n = hp.get('td_n', 1)
    
    def reinforce(self, experiences: MutableSequence[Experience], new_experiences: int):
        logging.info(f"reinforce: {len(experiences)} experiences, {new_experiences} new experiences")
        if len(experiences) < self._td_n:
            return
        sample = range(len(experiences) - self._td_n + 1)
        mini_batch = [[experiences[k] for k in range(i, i + self._td_n)] for i in sample]
        self.reinforce_actor(mini_batch)
        self._total_updates += 1
        experiences.clear()
    
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

    print(f"Creating {__name__} for {envId} with actor layers: {hp['layers']}, critic layers: {hp['c_layers']}, gamma: {hp['gamma']}, alpha: {hp['lr']}, beta: {hp['beta']}, td_n: {hp['td_n']}")
    
    return ActorCriticTDAgent(s_size, a_size, hp=hp)

def load_agent(env: gym.Env, state: Dict[str, Any]) -> Agent:
    envId = env.spec.id # type: ignore
    s_size = env.observation_space.shape[0] # type: ignore
    a_size = env.action_space.n # type: ignore

    hp = hyperparameters.get(envId.split('-')[0], hyperparameters['default']) | state.get('hp', {})
    
    agent = ActorCriticTDAgent(s_size, a_size, hp=hp)
    agent.load_state_dict(state)
    
    print(f"Loading {__name__} for {envId} with layers: {hp['layers']}, critic layers: {hp['c_layers']}, gamma: {agent._gamma}, alpha: {hp['lr']}, beta: {hp['beta']}, total_updates: {agent._total_updates}, td_n: {agent._td_n}") 

    return agent

if __name__ == "__main__":
    create_agent(None, ['-h']) # type: ignore