from numpy.random import RandomState

from alns.State import State
from alns.stopping_criteria.StoppingCriterion import StoppingCriterion


class MaxIterations(StoppingCriterion):
    def __init__(self, max_iterations: int):
        """
        Criterion that stops after a maximum number of iterations.
        """
        if max_iterations < 0:
            raise ValueError("Max iterations must be non-negative.")

        self._max_iterations = max_iterations
        self._current_iteration = 0

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def __call__(self, rnd: RandomState, best: State, current: State) -> bool:
        self._current_iteration += 1

        return self._current_iteration > self.max_iterations
