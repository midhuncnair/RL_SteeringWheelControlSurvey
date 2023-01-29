#! /usr/bin/env python3
"""
"""


# __all__ = []
__version__ = "1.0.0.0"
__author__ = "Midhun Chandran<midhunch@gmail.com>"
__maintainers__ = [
    "Midhun Chandran<midhunch@gmail.com>",
    "Vidyadhar<vidyadhar.bendre@gmail.com>",
    "Varghese<varghese.jacob4991@gmail.com>",
    "Mohana<mohana739@gmail.com>",
]


from .qlearning import QLearningAgent
from .dqn import DQNAgent
from .ddqn import DDQNAgent
from .duelingddqn import DuelingDDQNAgent
