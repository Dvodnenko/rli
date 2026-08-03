import matplotlib.pyplot as plt

from .context import ExperimentContext, ASMethod, ASMethodKind
from .main import experiment
from .gradient import experiment as g_experiment


ctx_1 = ExperimentContext(
    asmethod=None,
    alpha=0.1
)
ctx_2 = ExperimentContext(
    asmethod=None,
    alpha=0.4
)
ctx_3 = ExperimentContext(
    asmethod=None,
    alpha=0.1,
    mean_baseline=False,
    constant_baseline=0
)
ctx_4 = ExperimentContext(
    asmethod=None,
    alpha=0.4,
    mean_baseline=False,
    constant_baseline=0
)


# curve_1 = experiment(ctx_1)
# curve_2 = experiment(ctx_2)
curve_1 = g_experiment(ctx_1)
curve_2 = g_experiment(ctx_2)
curve_3 = g_experiment(ctx_3)
curve_4 = g_experiment(ctx_4)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

ax1.plot(curve_1[0], color="blue")
ax1.plot(curve_2[0], color="blue")
ax1.plot(curve_3[0], color="brown")
ax1.plot(curve_4[0], color="brown")
ax1.set_ylabel("Average reward")

ax2.plot(curve_1[1], color="blue")
ax2.plot(curve_2[1], color="blue")
ax2.plot(curve_3[1], color="brown")
ax2.plot(curve_4[1], color="brown")
ax2.set_ylabel("% Optimal action")
ax2.set_xlabel("Steps")

plt.tight_layout()
plt.show()
