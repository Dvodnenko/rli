from dataclasses import dataclass
from enum import Enum


class ASMethodKind(Enum):
    EPSILON = "epsilon" # greedy and epsilon-greedy

@dataclass(kw_only=True)
class ASMethod:
    """Action Selection Method"""
    kind: ASMethodKind
    epsilon: float = None

    def __post_init__(self):
        if self.kind == ASMethodKind.EPSILON:
            if not (0 <= self.epsilon <= 1):
                raise ValueError("Epsilon must be in range [0, 1]")


@dataclass(frozen=True, kw_only=True)
class ExperimentContext:
    asmethod: ASMethod # determines method selecting actions

    n_runs: int = 2000 # number of runs in the experiment
    n_steps: int = 1000 # number of steps in each run

    n_arms: int = 10 # number of arms
    drift_std: float = 0 # how much each arm's value drifts per step (for NSP)

    alpha: float | None = None # constant step-size parameter
    optimism: float = 0 # regulates initial optimism; realistic by default

    def __post_init__(self):
        """Validation of parameters"""
        if min(self.n_runs, self.n_steps, self.n_arms) <= 0:
            raise ValueError("N of runs, steps or arms can't be <= 0")

        if self.alpha is not None:
            if not (0 < self.alpha <= 1):
                raise ValueError("Alpha must be in range (0, 1]")
