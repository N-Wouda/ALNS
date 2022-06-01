from collections import deque
from typing import List, Optional

from alns.accept.AcceptanceCriterion import AcceptanceCriterion


class LateAcceptanceHillClimbing(AcceptanceCriterion):
    """
    Late acceptance hill climbing (LAHC) only accepts solutions that are better
    than the current solution from a specified number of iterations ago.

    Parameters
    ----------
    n_past: int
        A non-negative integer that specifies the number of last current
        solutions that need to be stored.
    improved: bool
        If True, use the Improved LAHC, which also accepts candidate solutions
        if they are better than the current solution. Default: False.
    """

    def __init__(self, n_past: int = 1, improved: bool = False):
        if not isinstance(n_past, int) or n_past < 1:
            raise ValueError("n_past argument must be a non-negative integer.")

        self.n_past = n_past
        self.past_objective: deque = deque([], maxlen=n_past)
        self.improved = improved

    def __call__(self, rnd, best, current, candidate):
        if not self.past_objectives:
            self.past_objectives.append(current.objective())
            return candidate.objective() <= current.objective()

        accept = candidate.objective() <= self.past_objectives[0]

        if self.improved:
            accept |= candidate.objective() <= current.objective()

        self.past_objectives.append(current.objective())

        return accept
