import time

from alns.State import State
from alns.stopping_criteria.StoppingCriterion import StoppingCriterion


class MaxRuntime(StoppingCriterion):
    def __init__(self, max_runtime: int) -> None:
        """
        Criterion that stops after a specified maximum runtime.
        """
        if max_runtime < 0:
            raise ValueError("Max runtime must be non-negative.")

        self._max_runtime = max_runtime
        self._elapsed_runtime = None
        self._start_runtime = None

    @property
    def max_runtime(self) -> float:
        return self._max_runtime

    @property
    def elapsed_runtime(self) -> float:
        return self._elapsed_runtime

    @property
    def start_runtime(self) -> float:
        """
        Reference point to calculate the elapsed time.
        """
        if self._start_runtime is None:
            self._start_runtime = time.perf_counter()

        return self._start_runtime

    def __call__(self, best: State, current: State) -> bool:
        # Reverse evaluation order to ensure that start_runtime is called first.
        self._elapsed_runtime = -(self.start_runtime - time.perf_counter())

        return self.elapsed_runtime > self.max_runtime
