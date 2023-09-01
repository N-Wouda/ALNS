from numpy.random import Generator

from alns.State import State


class MaxIterations:
    """
    Criterion that stops after a maximum number of iterations.
    """

    def __init__(self, max_iterations: int):
        if max_iterations < 0:
            raise ValueError("max_iterations < 0 not understood.")

        self._max_iterations = max_iterations
        self._current_iteration = 0

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def __call__(self, rng: Generator, best: State, current: State) -> bool:
        self._current_iteration += 1

        return self._current_iteration > self.max_iterations
