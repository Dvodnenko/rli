import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Bandit:
    n: int # n of arms
    qs: np.ndarray # actual values of arms

    def pull(self, arm: int) -> float:
        return np.random.normal(self.qs[arm], 1)


@dataclass
class Agent:
    bandit: Bandit
    Qs: np.ndarray
    N: np.ndarray
    epsilon: float = 0 # greedy by default

    def select_action(self):
        """return index of a greedy action or an exploratory action"""
        if random.random() <= self.epsilon:
            return random.randint(0, self.bandit.n-1) # random action
        return self.Qs.argmax() # greedy action

    def update(self, arm: int, reward: float):
        self.N[arm] += 1
        self.Qs[arm] += 1/self.N[arm] * (reward - self.Qs[arm])

    def step(self) -> tuple[int, float]:
        """Agent's whole action-reward cycle"""
        action = self.select_action()
        reward = self.bandit.pull(action)
        self.update(action, reward)
        return action, reward


def run(n_steps: int, n: int, epsilon: float):
    qs = np.random.normal(0, 1, size=n)
    Qs = np.zeros(n)
    N = np.zeros(n)
    rewards = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps)
    optimal_action = qs.argmax()

    bandit = Bandit(n, qs)
    agent = Agent(bandit, Qs, N, epsilon)

    for s in range(n_steps):
        step = agent.step()
        rewards[s] = step[1] # step[1] is the reward
        if step[0] == optimal_action: # step[0] is the action taken
            optimal_actions[s] += 1
    return rewards, optimal_actions


def experiment(n_runs=2000, n_steps=1000, n: int= 10, epsilon: float = 0):
    all_rewards = np.zeros((n_runs, n_steps))
    optimal_actions = np.zeros((n_runs, n_steps))
    
    for r in range(n_runs):
        all_rewards[r], optimal_actions[r] = run(n_steps, n, epsilon)
    
    avg_rewards = all_rewards.mean(axis=0)  # average across runs, per timestep
    return avg_rewards, optimal_actions.mean(axis=0) * 100


exp = experiment(n=10, epsilon=0.1)
# print(exp[0].shape, exp[1].shape)
# plt.subplot
# plt.xlabel("Steps")
# plt.ylabel("Average reward")
# plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

ax1.plot(exp[0])
ax1.set_ylabel("Average reward")

ax2.plot(exp[1])
ax2.set_ylabel("% Optimal action")
ax2.set_xlabel("Steps")

plt.tight_layout()
plt.show()
