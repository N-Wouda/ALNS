import time
from typing import Optional

from numpy.random import Generator

from alns.State import State


class MaxRuntime:
    """
    Criterion that stops after a specified maximum runtime.
    """

    def __init__(self, max_runtime: float):
        if max_runtime < 0:
            raise ValueError("max_runtime < 0 not understood.")

        self._max_runtime = max_runtime
        self._start_runtime: Optional[float] = None

    @property
    def max_runtime(self) -> float:
        return self._max_runtime

    def __call__(self, rng: Generator, best: State, current: State) -> bool:
        if self._start_runtime is None:
            self._start_runtime = time.perf_counter()

        return time.perf_counter() - self._start_runtime > self.max_runtime
