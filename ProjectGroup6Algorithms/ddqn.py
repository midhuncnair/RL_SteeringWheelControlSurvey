#! /usr/bin/env python3
"""
"""


__all__ = [
    "DDQNAgent",
]
__version__ = "1.0.0.0"
__author__ = "Mohana<mohana739@gmail.com>"
__maintainers__ = [
    "Midhun Chandran<midhunch@gmail.com>",
    "Vidyadhar<vidyadhar.bendre@gmail.com>",
    "Varghese<varghese.jacob4991@gmail.com>",
    "Mohana<mohana739@gmail.com>",
]


import random
import numpy as np
from .dqn import DQNAgent
from .utils import create_model


class DDQNAgent(DQNAgent):
    """
    """
    def __init__(
        self,
        env_name,
        *args,
        soft_update=False,
        tau=0.1, # target network soft update hyperparameter
        **kwargs
    ):
        """
        """
        super().__init__(env_name, *args, **kwargs)

        self.soft_update = soft_update
        self.tau = tau

        self.target_model = create_model(
            input_shape=(self.state_size,),
            action_space=self.action_size
        )

    @property
    def evaluation_model(self):
        """
        """
        return self.target_model

    def update_target_model(self):
        """after some time interval update the target model to be same with model
        """
        if not self.soft_update:
            self.target_model.set_weights(self.model.get_weights())
        else:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.tau) + q_weight * self.tau
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def replay(self):
        """
        """
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(self.batch_size, self.batch_size))

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
        target_val = self.target_model.predict(next_state, verbose=0)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # current Q Network selects the action
                # a'_max = argmax_a' Q(s', a')
                a = np.argmax(target_next[i])
                # target Q Network evaluates the action
                # Q_max = Q_target(s', a'_max)
                target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def when_done_for_run_episode(self, score, episode_id):
        """
        """
        super().when_done_for_run_episode(score, episode_id)
        self.update_target_model()
