#! /usr/bin/env python3
"""
"""


__all__ = [
    "QLearningAgent",
]
__version__ = "1.0.0.0"
__author__ = "Midhun Chandran<midhunch@gmail.com>"
__maintainers__ = [
    "Midhun Chandran<midhunch@gmail.com>",
    "Vidyadhar<vidyadhar.bendre@gmail.com>",
    "Varghese<varghese.jacob4991@gmail.com>",
    "Mohana<mohana739@gmail.com>",
]


import os
import random
import numpy as np
import pandas as pd
from .base import BaseAgent


class QLearningAgent(BaseAgent):
    """
    """
    def __init__(
        self,
        *args,
        alpha=0.1,
        gamma=0.6,
        epsilon=0.1,
        **kwargs
    ):
        """
        """
        super().__init__("Taxi-v3", *args, gamma=gamma, epsilon=epsilon, **kwargs)

        self.alpha = alpha
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

        self.q_table = np.zeros((self.state_size, self.action_size))

    @property
    def model_name(self):
        """
        """
        return '%s_.npy' % (self.env_name)

    def model_summary(self):
        """
        """
        return pd.DataFrame(
            self.q_table,
            index=['state_%i' % i for i in range(self.state_size)],
            columns=['action_%i' % i for i in range(self.action_size)]
        )

    def act(self, state):
        """
        """

        if np.random.random() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = np.argmax(self.q_table[state])

        return action

    def _load_model(self, model_path):
        """
        """
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.q_table = np.load(f)
        else:
            print("Load Model: model file at '%s' doesn't exist" % (model_path))

    def _save_model(self, model_path):
        """
        """
        print("Saving trained model to", model_path)
        with open(model_path, 'wb') as f:
            np.save(f, self.q_table)

    def when_done_for_run_episode(self, score, episode_id):
        """
        """
        average = self.plot_model(score, episode_id)
        self.save_model()
        print(
            "episode: {}/{}, score: {}, e: {:.2}, average: {}".format(
                episode_id, self.total_episodes, score, self.epsilon, average
            )
        )

    def run_episode(self, episode_id):
        """
        """
        state = self.env.reset()
        score, penalties, reward = 0, 0, 0
        done = False
        while not done:
            # self.env.render()
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)

            old_value = self.q_table[state, action]
            next_max = np.max(self.q_table[next_state])

            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

            self.q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            score += 1
            if done:
                self.when_done_for_run_episode(score, episode_id)

    def test(self, *args, **kwargs):
        """
        """
        self._total_test_scores, self._total_test_penalties = 0, 0
        super().test(*args, **kwargs)

        print("The results after running {} episodes:".format(self.total_test_episodes))
        print("\tAverage Score per episode is {}".format(self._total_test_scores/self.total_test_episodes))
        print("\tAverage Penalties per episode is {}".format(self._total_test_penalties/self.total_test_episodes))

    def test_episode(self, episode_id):
        """
        """
        state = self.env.reset()
        score, penalties, reward = 0, 0, 0
        done = False
        while not done:
            # self.env.render()
            action = np.argmax(self.q_table[state])
            state, reward, done, _ = self.env.step(action)
            if reward == -10:
                penalties += 1
            score += 1
            if done:
                print(
                    "episode: {}/{}, score: {}".format(
                        episode_id, self.total_test_episodes, score
                    )
                )
                break

        print("The score and penalty for this episode is {} and {}".format(score, penalties))
        self._total_test_penalties += penalties
        self._total_test_scores += score
