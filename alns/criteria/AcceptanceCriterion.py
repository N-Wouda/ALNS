from abc import ABC, abstractmethod

from ..State import State  # pylint: disable=unused-import
from numpy.random import RandomState  # pylint: disable=unused-import


class AcceptanceCriterion(ABC):
    """
    Base class from which to implement an acceptance criterion.
    """

    @abstractmethod
    def accept(self, rnd, best, current, candidate):
        """
        Determines whether to accept the proposed, candidate solution based on
        this acceptance criterion and the other solution states.

        Parameters
        ----------
        rnd : RandomState
            May be used to draw random numbers from.
        best : State
            The best solution state observed so far.
        current : State
            The current solution state.
        candidate : State
            The proposed solution state.

        Returns
        -------
        bool
            Whether to accept the candidate state (True), or not (False).
        """
        return NotImplemented
