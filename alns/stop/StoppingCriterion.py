from typing import Protocol

from numpy.random import Generator

from alns.State import State


class StoppingCriterion(Protocol):
    """
    Protocol describing a stopping criterion.
    """

    def __call__(self, rng: Generator, best: State, current: State) -> bool:
        """
        Determines whether to stop.

        Parameters
        ----------
        rng
            May be used to draw random numbers from.
        best
            The best solution state observed so far.
        current
            The current solution state.

        Returns
        -------
        bool
            Whether to stop iterating (True), or not (False).
        """
        ...  # pragma: no cover
