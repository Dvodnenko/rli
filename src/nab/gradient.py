import random
from dataclasses import dataclass

import numpy as np

from .context import ExperimentContext


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
        point = random.uniform(0, 1)
        cumulative = 0.0
        for a in range(self.bandit.n):
            cumulative += self.pi_t(a)
            if point <= cumulative:
                return a

    def update(self, action: int, reward: float):
        a = self.alpha
        Rt = reward
        Rt_bar = self.rewards.mean()
        pi_t_A = self.pi_t(action)

        self.preferences[action] += a*(Rt - Rt_bar)*(1 - pi_t_A)

        for action_ in range(self.bandit.n):
            if action_ != action:
                self.preferences[action_] -= a*(Rt - Rt_bar)*self.pi_t(action_)

    def step(self):
        """Agent's whole action-reward cycle"""
        action = self.select_action()
        reward = self.bandit.pull(action)
        self.update(action, reward)
        return action, reward


def run(ctx: ExperimentContext):
    actual_values = np.random.normal(0, 1, size=ctx.n_arms)
    preferences = np.zeros(ctx.n_arms)
    rewards = np.zeros(ctx.n_steps)
    optimal_actions = np.zeros(ctx.n_steps)

    bandit = Bandit(ctx.n_arms, actual_values, ctx.drift_std)
    agent = Agent(bandit, preferences, ctx.alpha, rewards)

    for s in range(ctx.n_steps):
        step = agent.step()
        rewards[s] = step[1] # step[1] is the reward
        if step[0] == actual_values.argmax(): # step[0] is the action taken
            optimal_actions[s] += 1
    return rewards, optimal_actions

def experiment(ctx: ExperimentContext):
    all_rewards = np.zeros((ctx.n_runs, ctx.n_steps))
    optimal_actions = np.zeros((ctx.n_runs, ctx.n_steps))
    
    for r in range(ctx.n_runs):
        all_rewards[r], optimal_actions[r] = run(ctx)
    
    avg_rewards = all_rewards.mean(axis=0)  # average across runs, per timestep
    return avg_rewards, optimal_actions.mean(axis=0) * 100
