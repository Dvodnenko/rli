import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Bandit:
    n: int
    actual_values: np.ndarray
    drift_std: float = 0  # how much each arm's value wobbles per step

    def pull(self, arm: int) -> float:
        value = np.random.normal(self.actual_values[arm], 1)
        self.actual_values += np.random.normal(0, self.drift_std, size=self.n)  # random walk, every step
        return value


@dataclass
class Agent:
    bandit: Bandit
    preferences: np.ndarray
    alpha: float
    rewards: np.ndarray

    def pi_t(self, a: int):
        """π_t(a) - the probability of picking action a"""
        es = np.zeros(self.bandit.n) + np.e # array of length n, full of numbers e
        pi_t_a = (np.e**self.preferences[a])/((es**self.preferences).sum())
        return pi_t_a

    def select_action(self):
        ...

    def update(self, action: int, reward: float):
        a = self.alpha
        Rt = reward
        Rt_bar = self.rewards.mean()
        pi_t_A = self.pi_t(action)

        self.preferences[action] += a*(Rt - Rt_bar)*(1 - pi_t_A)

        for action_ in range(self.bandit.n):
            if action_ != action:
                self.preferences[action_] -= a*(Rt - Rt_bar)*self.pi_t(action_)
