from typing import Optional

from numpy.random import Generator

from alns.State import State


class NoImprovement:
    """
    Criterion that stops if the best solution has not been improved
    after a number of iterations.

    Parameters
    ----------
    max_iterations
        The maximum number of non-improving iterations.
    """

    def __init__(self, max_iterations: int):
        if max_iterations < 0:
            raise ValueError("max_iterations < 0 not understood.")

        self._max_iterations = max_iterations
        self._target: Optional[float] = None
        self._counter = 0

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def __call__(self, rng: Generator, best: State, current: State) -> bool:
        if self._target is None or best.objective() < self._target:
            self._target = best.objective()
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self.max_iterations
