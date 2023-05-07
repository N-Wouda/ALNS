from typing import Protocol

from numpy.random import RandomState

from alns.State import State


class StoppingCriterion(Protocol):
    """
    Protocol describing a stopping criterion.
    """

    def __call__(self, rnd: RandomState, best: State, current: State) -> bool:
        """
        Determines whether to stop.

        Parameters
        ----------
        rnd
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
