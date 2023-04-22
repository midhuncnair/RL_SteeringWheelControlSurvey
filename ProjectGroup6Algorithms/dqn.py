#! /usr/bin/env python3
"""
"""


__all__ = [
    "DQNAgent",
]
__version__ = "1.0.0.0"
__author__ = "Varghese<varghese.jacob4991@gmail.com>"
__maintainers__ = [
    "Midhun Chandran<midhunch@gmail.com>",
    "Vidyadhar<vidyadhar.bendre@gmail.com>",
    "Varghese<varghese.jacob4991@gmail.com>",
    "Mohana<mohana739@gmail.com>",
]


import random
import numpy as np
from .base import BaseAgent
from .utils import create_model
from collections import deque
# from gym.utils import play


class DQNAgent(BaseAgent):
    """
    """
    def __init__(
        self,
        env_name,
        *args,
        batch_size=64,
        train_start=1000,
        memory_size=2000,
        **kwargs
    ):
        """
        """
        super().__init__(env_name, *args, **kwargs)

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.batch_size = batch_size
        self.train_start = train_start
        self.memory_size = memory_size

        self.memory = deque(maxlen=self.memory_size)
        self.model = create_model(
            input_shape=(self.state_size,),
            action_space=self.action_size
        )

    @property
    def evaluation_model(self):
        return self.model

    def model_summary(self):
        """
        """
        self.model.summary()

    def remember(self, state, action, reward, next_state, done):
        """
        """
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        """
        """
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.evaluation_model.predict(state, verbose=0))

    def replay(self):
        """
        """
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state, verbose=0)
        target_next = self.model.predict(next_state, verbose=0)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

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
        state = np.reshape(state, [1, self.state_size])
        done = False
        score = 0
        while not done:
            self.env.render()
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.reshape(next_state, [1, self.state_size])
            if not done or score == self.env._max_episode_steps-1:
                reward = reward
            else:
                reward = -100
            self.remember(state, action, reward, next_state, done)
            state = next_state
            score += 1
            if done:
                self.when_done_for_run_episode(score, episode_id)
            self.replay()

    def test_episode(self, episode_id):
        """
        """
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        done = False
        score = 0
        while not done:
            self.env.render()
            action = np.argmax(self.evaluation_model.predict(state, verbose=0))
            next_state, reward, done, _ = self.env.step(action)
            state = np.reshape(next_state, [1, self.state_size])
            score += 1
            if done:
                print(
                    "episode: {}/{}, score: {}".format(
                        episode_id, self.total_test_episodes, score
                    )
                )
                break
