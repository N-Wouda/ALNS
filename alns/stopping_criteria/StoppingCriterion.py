from abc import ABC, abstractmethod

from numpy.random import RandomState

from alns.State import State


class StoppingCriterion(ABC):
    """
    Base class from which to implement a stopping criterion.
    """

    @abstractmethod
    def __call__(self) -> bool:
        """
        Determines whether to stop based on the implemented stopping criterion.

        Returns
        -------
        Whether to stop the iteration (True), or not (False).
        """
        return NotImplemented
