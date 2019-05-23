import gym
from gym import spaces
import random
import numpy as np


class RenderEnv(gym.ObservationWrapper):
    def observation(self, observation):
        self.render()
        return observation


class PermuteEnv(gym.ObservationWrapper):
    def __init__(self, env, axes):
        super(PermuteEnv, self).__init__(env)

        self.axes = axes

        new_shape = []
        for axis in axes:
            new_shape.append(env.observation_space.shape[axis])
        new_shape = tuple(new_shape)
        self.observation_space = spaces.Box(low=np.min(env.observation_space.low), high=np.max(env.observation_space.high), dtype=env.observation_space.dtype, shape=new_shape)

    def observation(self, observation):
        return np.transpose(observation, self.axes)


class MapEnv(gym.ObservationWrapper):
    def __init__(self, env, m):
        super(MapEnv, self).__init__(env)
        self.m = m

        def func(x):
            return self.m[x]
        self.vec_func = np.vectorize(func)

    def observation(self, observation):
        return self.vec_func(observation)


class SampleEnv(gym.Env):
    def __init__(self, envs):
        super(SampleEnv, self).__init__()
        self.envs = envs
        self._cur_env = None
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.reward_range = self.envs[0].reward_range

    def reset(self):
        self._cur_env = random.choice(self.envs)
        return self._cur_env.reset()

    def step(self, action):
        return self._cur_env.step(action)

    def render(self, *args):
        self._cur_env.render(*args)


def play(env):
    env = RenderEnv(env)
    env.reset()
    done = False
    while not done:
        action = int(input())
        env.step(action)
