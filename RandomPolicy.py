
from agent import Agent, Action
import gymnasium as gym
import random
import torch
from typing import List, Dict, Any

class RandomAgent(Agent):
    def __init__(self, a_size: int):
        """Initialize the random policy agent with the action space size.
        Args:
            a_size (int): The size of the action space.
        """
        super(RandomAgent, self).__init__()
        self._a_size = a_size
 
    def act(self, state: torch.Tensor) -> Action:
        """Select an action from the action space base on the current state.
        Args:
            state (torch.Tensor): The current state of the environment.
        Returns:
            Action: A tuple containing the selected action
            and optional tensor for traning.
        """
        return Action(torch.tensor(random.randint(0, self._a_size - 1)), None, None)
    
    def get_state_dict(self) -> Dict[str, Any]:
        return {
            'a_size': self._a_size,
        }


def create_agent(env: gym.Env, parameters: List[str]) -> Agent:
    """Create a random policy agent for the given environment.
    Args:
        env (gym.Env): The environment to create the agent for.
        parameters (List[str]): Additional parameters for the agent.
    Returns:
        Agent: The created random policy agent.
    """
    print(f"Creating RandomPolicy for {env.spec.id}") # type: ignore
    return RandomAgent(env.action_space.n) # type: ignore

def load_agent(env: gym.Env, state: Dict[str, Any]) -> Agent:
    """Load a random policy agent from the given state.
    Args:
        env (gym.Env): The environment to load the agent for.
        state (Dict[str, Any]): The state dictionary of the agent.
    Returns:
        Agent: The loaded random policy agent of output.
    """    
    print(f"Loading RandomPolicy for {env.spec.id}") # type: ignore
    assert state['a_size'] == env.action_space.n, "action space size mismatch" # type: ignore
    return RandomAgent(env.action_space.n) # type: ignore
