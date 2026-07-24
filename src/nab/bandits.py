import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


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
    alpha: float = 0.1 # step-size

    def select_action(self):
        """return index of a greedy action or an exploratory action"""
        if random.random() < self.epsilon:
            return random.randint(0, self.bandit.n-1) # random action
        return self.Qs.argmax() # greedy action

    def update(self, arm: int, reward: float):
        self.N[arm] += 1
        # self.Qs[arm] += 1/self.N[arm] * (reward - self.Qs[arm])
        self.Qs[arm] += self.alpha * (reward - self.Qs[arm])

    def step(self) -> tuple[int, float]:
        """Agent's whole action-reward cycle"""
        action = self.select_action()
        reward = self.bandit.pull(action)
        self.update(action, reward)
        return action, reward


def run(
    n_steps: int,
    n: int, drift_std: float = 0,
    epsilon: float = 0, alpha: float = 0.1
):
    qs = np.random.normal(0, 1, size=n)
    Qs = np.zeros(n)
    N = np.zeros(n)
    rewards = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps)

    bandit = Bandit(n, qs, drift_std)
    agent = Agent(bandit, Qs, N, epsilon, alpha)

    for s in range(n_steps):
        step = agent.step()
        rewards[s] = step[1] # step[1] is the reward
        if step[0] == qs.argmax(): # step[0] is the action taken
            optimal_actions[s] += 1
    return rewards, optimal_actions


def experiment(
    n_runs=2000, n_steps=1000,
    n: int = 10, drift_std: float = 0,
    epsilon: float = 0, alpha: float = 0.1
):
    all_rewards = np.zeros((n_runs, n_steps))
    optimal_actions = np.zeros((n_runs, n_steps))
    
    for r in range(n_runs):
        all_rewards[r], optimal_actions[r] = run(n_steps, n, drift_std, epsilon, alpha)
    
    avg_rewards = all_rewards.mean(axis=0)  # average across runs, per timestep
    return avg_rewards, optimal_actions.mean(axis=0) * 100


curve_1 = experiment(n=10, epsilon=0.1, drift_std=0.1, alpha=0.5)
curve_2 = experiment(n=10, epsilon=0.01, drift_std=0.1, alpha=0.5)
curve_3 = experiment(n=10, epsilon=0, drift_std=0.1, alpha=0.5)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

ax1.plot(curve_1[0], color="black")
ax1.plot(curve_2[0], color="red")
ax1.plot(curve_3[0], color="green")
ax1.set_ylabel("Average reward")

ax2.plot(curve_1[1], color="black")
ax2.plot(curve_2[1], color="red")
ax2.plot(curve_3[1], color="green")
ax2.set_ylabel("% Optimal action")
ax2.set_xlabel("Steps")

plt.tight_layout()
plt.show()
