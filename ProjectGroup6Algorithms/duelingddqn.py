#! /usr/bin/env python3
"""
"""


__all__ = [
    "DuelingDDQNAgent",
]
__version__ = "1.0.0.0"
__author__ = "Vidyadhar<vidyadhar.bendre@gmail.com>"
__maintainers__ = [
    "Midhun Chandran<midhunch@gmail.com>",
    "Vidyadhar<vidyadhar.bendre@gmail.com>",
    "Varghese<varghese.jacob4991@gmail.com>",
    "Mohana<mohana739@gmail.com>",
]


from .ddqn import DDQNAgent
from .utils import create_model


class DuelingDDQNAgent(DDQNAgent):
    """
    """
    def __init__(
        self,
        env_name,
        *args,
        **kwargs
    ):
        """
        """
        super().__init__(env_name, *args, **kwargs)

        self.model = create_model(
            input_shape=(self.state_size,),
            action_space=self.action_size,
            dueling=True,
        )

        self.target_model = create_model(
            input_shape=(self.state_size,),
            action_space=self.action_size,
            dueling=True,
        )
