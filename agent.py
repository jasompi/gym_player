import collections
import gymnasium as gym
import numpy as np
import torch
import torch.types as torch_types
from typing import List, Dict, NamedTuple, Tuple, Any, Optional

class Experience(NamedTuple):
    """Experience tuple for reinforcement learning.
    Attributes:
        state (np.ndarray): The current state of the environment.
        action (int): The action taken.
        reward (float): The reward received.
        new_state (np.ndarray): The new state after taking the action.
        done (bool): Whether the episode has ended.
        log_prob (torch.Tensor): The log probability of the action taken.
    """
    state: np.ndarray
    action: torch_types.Number
    reward: float
    new_state: np.ndarray
    done: bool
    extra: Optional[torch.Tensor]

class Agent:
    def __init__(self):
        """Initialize thepolicy agent with the action space size."""
        pass
    
    def train(self, train: bool):
        """Set the agent to training mode.
        Args:
            train (bool): If True, set the agent to training mode.
        """
        pass
    
    def act(self, state: np.ndarray) -> Tuple[torch_types.Number, Optional[torch.Tensor]]:
        """Select an action from the action space base on the current state.
        Args:
            state (np.ndarray): The current state of the environment.
        Returns:
            Tuple[torch_types.Number, Any]: A tuple containing the selected action
            and optional tensor for traning.
        """
        return 0, None
    
    def reinforce(self, experiences: collections.deque[Experience]):
        """Perform reinforcement learning update.
        Args:
            experiences (collections.deque[Experience]): A deque of experiences to update the agent.
        """
        pass
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get the state dictionary of the agent.
        Returns:
            Dict[str, Any]: The state dictionary of the agent.
        """
        return {}
