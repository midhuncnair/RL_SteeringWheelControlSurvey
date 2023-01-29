#! /usr/bin/env python3
"""
"""


__all__ = [
    'create_model',
    'thread_pool',
]
__version__ = "1.0.0.0"
__author__ = "Midhun Chandran<midhunch@gmail.com>"
__maintainers__ = [
    "Midhun Chandran<midhunch@gmail.com>",
    "Vidyadhar<vidyadhar.bendre@gmail.com>",
    "Varghese<varghese.jacob4991@gmail.com>",
    "Mohana<mohana739@gmail.com>",
]

from functools import wraps
from concurrent.futures import ThreadPoolExecutor

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras.optimizers import (
    RMSprop
)
from tensorflow.keras import backend as keras_backend


_DEFAULT_POOL_EXECUTER = ThreadPoolExecutor()


def create_model(input_shape, action_space, dueling=False):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    if dueling:
        state_value = Dense(1, kernel_initializer='he_uniform')(X)
        state_value = Lambda(lambda s: keras_backend.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

        action_advantage = Dense(action_space, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - keras_backend.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)

        X = Add()([state_value, action_advantage])
    else:
        # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='CartPoleDuelingDDQNmodel')
    model.compile(loss="mean_squared_error", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    # model.summary()
    return model



def thread_pool(func):
    """The thread pool that communicats back to the main
    thread used ThreadPoolExecuter.
    """
    @wraps(func)
    def inner(*args, **kwargs):
        """
        """
        return _DEFAULT_POOL_EXECUTER.submit(func, *args, **kwargs)

    return inner
