from collections import deque
from typing import List, Optional

from alns.accept.AcceptanceCriterion import AcceptanceCriterion


class LateAcceptanceHillClimbing(AcceptanceCriterion):
    """
    Late acceptance hill climbing only accepts solutions that are better than the
    current solution from a specified number of iterations ago.

    Parameters
    ----------
    # TODO improve doc string
    - Late Acceptance Hill Climbing: history_size >= 1
    - Improved Late Acceptance Hill Climbing: history_size >=1, compare_current=True
    """

    def __init__(self, history_size: int = 1, improved: bool = False):
        if not isinstance(history_size, int) or history_size < 1:
            raise ValueError(
                "History size argument must be a non-negative integer."
            )

        self.history_size = history_size
        self.history: deque = deque([], maxlen=history_size)
        self.improved = improved

    def __call__(self, rnd, best, current, candidate):
        if not self.history:
            self.history.append(current.objective())
            return candidate.objective() <= current.objective()

        accept = candidate.objective() <= self.history[0]

        if self.improved:
            accept |= candidate.objective() <= current.objective()

        self.history.append(current.objective())

        return accept
