from abc import ABC, abstractmethod

from numpy.random import RandomState

from alns.State import State


class StoppingCriterion(ABC):
    """
    Base class from which to implement a stopping criterion.
    """

    @abstractmethod
    def __call__(self, best: State, current: State) -> bool:
        """
        Determines whether to stop based on the implemented stopping criterion.

        Parameters
        ----------
        best
            The best solution state observed so far.
        current
            The current solution state.

        Returns
        -------
        Whether to stop the iteration (True), or not (False).
        """
        return NotImplemented
