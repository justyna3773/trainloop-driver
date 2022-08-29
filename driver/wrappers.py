import gym
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None, penalty=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._penalty = penalty

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        if self._penalty is not None:
            reward = self._penalty
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)