import random
from dataclasses import dataclass

import numpy as np

from .context import ExperimentContext


@dataclass
class Bandit:
    n: int
    qs: np.ndarray
    drift_std: float = 0  # how much each arm's value wobbles per step

    def pull(self, arm: int) -> float:
        value = np.random.normal(self.qs[arm], 1)
        self.qs += np.random.normal(0, self.drift_std, size=self.n)  # random walk, every step
        return value


@dataclass
class Agent:
    bandit: Bandit
    Qs: np.ndarray
    N: np.ndarray
    epsilon: float = 0 # greedy by default
    alpha: float | None = None # step-size

    def select_action(self):
        """return index of a greedy action or an exploratory action"""
        if random.random() < self.epsilon:
            return random.randint(0, self.bandit.n-1) # random action
        return self.Qs.argmax() # greedy action

    def update(self, arm: int, reward: float):
        self.N[arm] += 1
        # self.Qs[arm] += 1/self.N[arm] * (reward - self.Qs[arm])
        step_size = self.alpha if self.alpha is not None else 1/self.N[arm]
        self.Qs[arm] += step_size * (reward - self.Qs[arm])

    def step(self) -> tuple[int, float]:
        """Agent's whole action-reward cycle"""
        action = self.select_action()
        reward = self.bandit.pull(action)
        self.update(action, reward)
        return action, reward


def run(ctx: ExperimentContext):
    qs = np.random.normal(0, 1, size=ctx.n_arms)
    Qs = np.zeros(ctx.n_arms) + ctx.optimism
    N = np.zeros(ctx.n_arms)
    rewards = np.zeros(ctx.n_steps)
    optimal_actions = np.zeros(ctx.n_steps)

    bandit = Bandit(ctx.n_arms, qs, ctx.drift_std)
    agent = Agent(bandit, Qs, N, ctx.epsilon, ctx.alpha)

    for s in range(ctx.n_steps):
        step = agent.step()
        rewards[s] = step[1] # step[1] is the reward
        if step[0] == qs.argmax(): # step[0] is the action taken
            optimal_actions[s] += 1
    return rewards, optimal_actions


def experiment(ctx: ExperimentContext):
    all_rewards = np.zeros((ctx.n_runs, ctx.n_steps))
    optimal_actions = np.zeros((ctx.n_runs, ctx.n_steps))
    
    for r in range(ctx.n_runs):
        all_rewards[r], optimal_actions[r] = run(ctx)
    
    avg_rewards = all_rewards.mean(axis=0)  # average across runs, per timestep
    return avg_rewards, optimal_actions.mean(axis=0) * 100
