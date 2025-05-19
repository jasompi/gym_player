import torch
from typing import Dict, MutableSequence, NamedTuple, Any, Optional

class Experience(NamedTuple):
    """Experience tuple for reinforcement learning.
    Attributes:
        state (np.ndarray): The current state of the environment.
        action (torch_types.Number | torch.Tensor): The action taken.
        reward (float): The reward received.
        done (bool): Whether the episode has ended.
        next_state (np.ndarray | None): The next state if the episode was truncated, 
            other wise None and the next state is in the next experience.
        log_prob (torch.Tensor): The log probability of the action taken.
        value (torch.Tensor): The extimated value for the state.
    """
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    done: bool
    truncated: bool
    next_state: torch.Tensor
    next_action: torch.Tensor
    log_prob: Optional[torch.Tensor]
    value: Optional[torch.Tensor]
    next_value: Optional[torch.Tensor]


class Action(NamedTuple):
    """
    Action: NamedTuple containing information about an action taken by an agent.

    Attributes:
        action (torch_types.Number | torch.Tensor): The action taken by the agent. Can be a single number or a tensor.
        log_prob (Optional[torch.Tensor]): The log probability of the action, if available.
        value (Optional[torch.Tensor]): The value associated with the action, if available.
    """
    action: torch.Tensor
    log_prob: Optional[torch.Tensor]
    value: Optional[torch.Tensor]

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
    
    def act(self, state: torch.Tensor) -> Action:
        """Select an action from the action space base on the current state.
        Args:
            state (torch.Tensor): The current state of the environment.
        Returns:
            Action: A tuple containing the selected action
            and optional tensor for traning.
        """
        return Action(action=0, log_prob=None, value=None)
    
    def reinforce(self, experiences: MutableSequence[Experience], new_experience: int):
        """Perform reinforcement learning update.
        Args:
            experiences (collections.deque[Experience]): A deque of experiences to update the agent.
            new_experience (int): The new experience added to the deque.
        """
        pass
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get the state dictionary of the agent.
        Returns:
            Dict[str, Any]: The state dictionary of the agent.
        """
        return {}
