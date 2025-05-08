import gymnasium as gym
import logging
import numpy as np
import random
import torch
import torch.types as torch_types
from typing import List, Dict, Tuple, Any, Optional

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
    
    def reinforce(self, replay_buffer: List[Tuple[np.ndarray, int, float, np.ndarray, torch.Tensor]]):
        """Perform reinforcement learning update.
        Args:
            replay_buffer (List[Tuple[np.ndarray, int, float, np.ndarray, torch.Tensor]]): The replay buffer
            containing the transitions to update the policy.
        """
        pass
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get the state dictionary of the agent.
        Returns:
            Dict[str, Any]: The state dictionary of the agent.
        """
        return {}
