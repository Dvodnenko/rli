import random

import numpy as np
import matplotlib.pyplot as plt

n = 10 # n of arms


def select_action(Qs, epsilon: float) -> int:
    """return index of a greedy action or an exploratory action"""
    if random.random() <= epsilon:
        return random.randint(0, n-1) # random action
    return Qs.argmax() # greedy action

def count(a: int, N):
    N[a] += 1
    return

def pull(a: int, qs) -> float:
    """simulate pulling arm a, returns a noisy reward"""
    return np.random.normal(qs[a], 1)

def average(a: int, reward: float, Qs, N):
    Qs[a] += 1/N[a] * (reward - Qs[a])


def run(n_steps=1000, epsilon: float = 0):
    qs = np.random.normal(0, 1, size=n)
    Qs = np.zeros(n)
    N = np.zeros(n)
    rewards = np.zeros(n_steps)
    
    for s in range(n_steps):
        a = select_action(Qs, epsilon)
        reward = pull(a, qs)
        count(a, N)
        average(a, reward, Qs, N)
        rewards[s] = reward
    
    return rewards


def experiment(n_runs=3000, n_steps=1000, epsilon: float = 0):
    all_rewards = np.zeros((n_runs, n_steps))
    
    for r in range(n_runs):
        all_rewards[r] = run(n_steps, epsilon)
    
    avg_rewards = all_rewards.mean(axis=0)  # average across runs, per timestep
    return avg_rewards


plt.plot(experiment(epsilon=0.1))
plt.plot(experiment(epsilon=0.01))
plt.show()
