#! /usr/bin/env python3
"""
"""


__all__ = [
    'BaseAgent'
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
import json
import gym
import pylab
from tensorflow.keras.models import load_model
from gym.wrappers import Monitor
# from .utils import thread_pool
# import pdb

SET_FIG_SIZE = False


class BaseAgent:
    """
    """
    def __init__(
        self,
        env_name,
        total_episodes=100,
        total_test_episodes=None,
        save_interval=10,
        gamma=0.95,  # discount rate
        epsilon=1.0,  # exploration rate
        epsilon_min=0.01,  # min exploration probability
        epsilon_decay=0.999,  # exponential decay rate for exploration prob
        video_save_base_path='Results/saved/vids/',
        model_save_base_path='Results/saved/models/',
        env_seed=0,
        env_max_episode_steps=4000,
    ):
        self.env_name = env_name
        self._base_env = None
        self.total_episodes = total_episodes
        self.total_test_episodes = (
            total_test_episodes
            if total_test_episodes is not None else
            self.total_episodes
        )
        self.save_interval = save_interval

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.video_save_base_path = os.path.join(
            video_save_base_path,
            self.__class__.__name__,
            self.env_name,
        )
        self.model_save_base_path = os.path.join(
            model_save_base_path,
            self.__class__.__name__,
        )
        self.model_save_path = os.path.join(
            self.model_save_base_path,
            self.model_name,
        )
        self.agent_save_path = os.path.join(
            self.model_save_base_path,
            self.agent_name,
        )
        os.makedirs(self.model_save_base_path, exist_ok=True)
        self.env_seed = env_seed
        self.env_max_episode_steps = env_max_episode_steps

        self._current_episode = None
        self._is_test = False

        self.scores, self.episodes, self.averages = [], [], []
        self._agent_data_keys = {'scores', 'episodes', 'averages'}

    def _init_env(self):
        """
        """
        self._base_env = gym.make(self.env_name)
        self._base_env.seed(self.env_seed)
        self._monitoring_env = self._init_monitoring_env()

        self._base_env._max_episode_steps = self.env_max_episode_steps
        self._monitoring_env._max_episode_steps = self.env_max_episode_steps

    def _init_monitoring_env(self):
        """
        """
        os.makedirs(self.video_save_base_path, exist_ok=True)
        return Monitor(
            self._base_env,
            self.video_save_base_path,
            video_callable=lambda episode_id: True,
            force=True
        )

    @property
    def env(self):
        """
        """
        if self._base_env is None:
            self._init_env()

        if (
            self._current_episode in [0, self.total_episodes - 1]
            or (
                self._current_episode is not None
                and self._current_episode % self.interval_mod_oper == 0
            )
        ):
            return self._monitoring_env
        else:
            return self._base_env

    @property
    def interval_mod_oper(self):
        """
        """
        return (
            self.total_test_episodes // self.save_interval
            if self._is_test else
            self.total_episodes // self.save_interval
        )

    @property
    def model_name(self):
        """
        """
        return '%s_.h5' % (self.env_name)

    @property
    def agent_name(self):
        """
        """
        return '%s_.json' % (self.env_name)

    def get_agent_data(self):
        """
        """
        data = {}
        for key in self._agent_data_keys:
            data[key] = getattr(self, key)
        return data

    def _load_model(self, model_path):
        """
        """
        if os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            print("Load Model: model file at '%s' doesn't exist" % (model_path))

    def _load_agent_data(self, agent_path):
        """
        """
        if os.path.exists(agent_path):
            with open(agent_path, 'r') as agent_file:
                data = json.load(agent_file)
                for key in self._agent_data_keys:
                    key_data = data.get(key, getattr(self, key))
                    setattr(self, key, key_data)
        else:
            print("Load Model: agent file at '%s' doesn't exist" % (agent_path))

    def load_model(self, model_path=None, agent_path=None):
        """
        """
        if model_path is None:
            model_path = self.model_save_path
        if agent_path is None:
            agent_path = self.agent_save_path
        self._load_model(model_path)
        self._load_agent_data(agent_path)

    def _save_model(self, model_path):
        """
        """
        print("Saving trained model to", model_path)
        self.model.save(model_path)

    def _save_agent_data(self, agent_path):
        """
        """
        print("Saving agent data to", agent_path)
        with open(agent_path, 'w') as agent_file:
                json.dump(self.get_agent_data(), agent_file)

    def save_model(self, model_path=None, agent_path=None, force=False):
        """
        """
        if model_path is None:
            model_path = self.model_save_path
        if agent_path is None:
            agent_path = self.agent_save_path
        if (
            force
            or (
                self._current_episode is not None
                and self._current_episode % self.interval_mod_oper == 0
            )
        ):
            self._save_model(model_path)
            self._save_agent_data(agent_path)

    def plot_model(self, score=None, episode=None):
        """
        """
        global SET_FIG_SIZE

        if score is not None and episode is not None:
            self.scores.append(score)
            self.episodes.append(episode)
            self.averages.append(sum(self.scores[-50:]) / len(self.scores[-50:]))

        else:
            if not SET_FIG_SIZE:
                pylab.figure(figsize=(18, 9))
                SET_FIG_SIZE = True
            pylab.plot(self.episodes, self.averages, 'r')
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)

        return str(self.averages[-1])[:5]

    def run_episode(self, episode_id):
        """
        """
        raise NotImplementedError("This needs to be implemented in the child class")

    def _close_env(self):
        """
        """
        self._base_env.reset()
        self._monitoring_env.reset()
        self._base_env.close()
        self._monitoring_env.close()

    # @thread_pool
    def run(self):
        """
        """
        try:
            self._is_test = False
            for e in range(1, self.total_episodes + 1):
                self._current_episode = e
                ret = self.run_episode(e)
                if ret is True:
                    return
        except Exception as err:
            print("FatalError and is %s:%s"%(err.__class__.__name__, str(err)))
        finally:
            self._close_env()

    def test_episode(self, episode_id):
        """
        """
        raise NotImplementedError("This needs to be implemented in the child class")

    # @thread_pool
    def test(self):
        """
        """
        try:
            self._is_test = True
            self.load_model()
            for e in range(1, self.total_test_episodes + 1):
                self._current_episode = e
                ret = self.test_episode(e)
                if ret is True:
                    return
        except Exception as err:
            print("FatalError and is %s:%s"%(err.__class__.__name__, str(err)))
        finally:
            self._close_env()
