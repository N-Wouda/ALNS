from numpy.random import RandomState
from typing import Optional

from alns.State import State
from alns.stop.StoppingCriterion import StoppingCriterion


class NoImprovement(StoppingCriterion):
    """
    Criterion that stops if the best solution has not been improved
    for a number of iterations.

    The first iteration is used to initialize the criterion. Hence,
    the algorithm will iterate at least n_iterations + 1 times.

    Parameters
    ----------
    n_iterations
        The maximum number of non-improving iterations.
    """

    def __init__(self, n_iterations: int):
        if n_iterations < 0:
            raise ValueError("n_iterations < 0 not understood.")

        self._n_iterations = n_iterations
        self._target: Optional[float] = None

    @property
    def n_iterations(self) -> int:
        return self._n_iterations

    def __call__(self, rnd: RandomState, best: State, current: State) -> bool:
        if self._target is None or best.objective() < self._target:
            self._target = best.objective()
            self._counter = 0
            return False

        self._counter += 1
        return self._counter > self.n_iterations
