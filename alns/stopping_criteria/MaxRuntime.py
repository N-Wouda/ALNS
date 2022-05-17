import time

from typing import Optional
from numpy.random import RandomState

from alns.State import State
from alns.stopping_criteria.StoppingCriterion import StoppingCriterion


class MaxRuntime(StoppingCriterion):
    def __init__(self, max_runtime: float):
        """
        Criterion that stops after a specified maximum runtime.
        """
        if max_runtime < 0:
            raise ValueError("Max runtime must be non-negative.")

        self._max_runtime = max_runtime
        self._start_runtime: Optional[float] = None

    @property
    def max_runtime(self) -> float:
        return self._max_runtime

    def __call__(self, rnd: RandomState, best: State, current: State) -> bool:
        if self._start_runtime is None:
            self._start_runtime = time.perf_counter()

        return time.perf_counter () - self._start_runtime > self.max_runtime
