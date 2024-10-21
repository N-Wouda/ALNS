from typing import Protocol

from numpy.random import Generator

from alns.State import State


class AcceptanceCriterion(Protocol):
    """
    Protocol describing an acceptance criterion.
    """

    def __call__(
        self, rng: Generator, best: State, current: State, candidate: State
    ) -> bool:
        """
        Determines whether to accept the proposed, candidate solution based on
        this acceptance criterion and the other solution states.

        Parameters
        ----------
        rng
            May be used to draw random numbers from.
        best
            The best solution state observed so far.
        current
            The current solution state.
        candidate
            The proposed solution state.

        Returns
        -------
        bool
            Whether to accept the candidate state (True), or not (False).
        """
        ...  # pragma: no cover
