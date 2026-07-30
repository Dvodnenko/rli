import matplotlib.pyplot as plt

from .context import ExperimentContext
from .main import experiment


ctx_1 = ExperimentContext(
    epsilon=0, alpha=0.1, optimism=5
)
ctx_2 = ExperimentContext(
    epsilon=0.1, alpha=0.1
)


curve_1 = experiment(ctx_1)
curve_2 = experiment(ctx_2)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

ax1.plot(curve_1[0], color="black")
ax1.plot(curve_2[0], color="red")
ax1.set_ylabel("Average reward")

ax2.plot(curve_1[1], color="black")
ax2.plot(curve_2[1], color="red")
ax2.set_ylabel("% Optimal action")
ax2.set_xlabel("Steps")

plt.tight_layout()
plt.show()
