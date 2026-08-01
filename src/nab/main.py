import random
from dataclasses import dataclass

import numpy as np

from .context import ExperimentContext, ASMethod, ASMethodKind


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
    estimated_values: np.ndarray
    N: np.ndarray
    asmethod: ASMethod
    alpha: float | None = None # step-size

    def select_action(self):
        if self.asmethod.kind == ASMethodKind.EPSILON:
            return self._select_action_epsilon()
        else:
            return self._select_action_ucb()

    def _select_action_epsilon(self):
        if random.random() < self.asmethod.epsilon:
            return random.randint(0, self.bandit.n-1) # random action
        return self.estimated_values.argmax() # greedy action

    def _select_action_ucb(self):
        """select action with UCB1 formula"""
        t = self.N.sum() + 1
        with np.errstate(divide='ignore'):
            ucb_values = self.estimated_values + self.asmethod.c * np.sqrt(np.log(t) / self.N)
        ucb_values[self.N == 0] = np.inf
        return np.argmax(ucb_values)

    def update(self, arm: int, reward: float):
        self.N[arm] += 1
        # self.estimated_values[arm] += 1/self.N[arm] * (reward - self.estimated_values[arm])
        step_size = self.alpha if self.alpha is not None else 1/self.N[arm]
        self.estimated_values[arm] += step_size * (reward - self.estimated_values[arm])

    def step(self) -> tuple[int, float]:
        """Agent's whole action-reward cycle"""
        action = self.select_action()
        reward = self.bandit.pull(action)
        self.update(action, reward)
        return action, reward


def run(ctx: ExperimentContext):
    actual_values = np.random.normal(0, 1, size=ctx.n_arms)
    estimated_values = np.zeros(ctx.n_arms) + ctx.optimism
    N = np.zeros(ctx.n_arms)
    rewards = np.zeros(ctx.n_steps)
    optimal_actions = np.zeros(ctx.n_steps)

    bandit = Bandit(ctx.n_arms, actual_values, ctx.drift_std)
    agent = Agent(bandit, estimated_values, N, ctx.asmethod, ctx.alpha)

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
