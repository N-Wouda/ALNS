from collections import deque
from typing import List, Optional

from alns.accept.AcceptanceCriterion import AcceptanceCriterion


class HillClimbing(AcceptanceCriterion):
    """
    The Hill Climbing (HC) criterion only accepts progressively better solutions,
    discarding those that result in a worse objective value.

    There are three variants of the HC criterion:
    1) classic (HC), which only accepts a candidate solution if it is better
       than the current solution.
    2) late-acceptance (LAHC), which only accepts a candidate solution if it is
       better than the current solution from a number of iterations ago.
    3) improved LA (ILAHC), which is the LA criterion but also accepts a
       candidate solution if it is better than the current solution.

    Parameters
    ----------
    on_current: bool
        A boolean to denote whether to accept candidate solutions if they are
        better than the current solution. Default: True.
    n_last: int
        A non-negative integer that specifies the number of last current
        solutions that need to be stored. Default: None, which means that no
        previous current solutions are stored.
    """

    def __init__(self, on_current: bool = True, n_last: Optional[int] = None):
        self._on_current = on_current

        if n_last is not None:
            if not isinstance(n_last, int) or n_last < 0:
                raise ValueError(
                    "n_last argument must be a non-negative integer."
                )

            self._n_last: int = n_last
            self._last_objectives: deque = deque([], maxlen=n_last)

    # TODO Add properties; how to add optional properties?

    def __call__(self, rnd, best, current, candidate):
        cand_obj = candidate.objective()
        curr_obj = current.objective()
        result = False

        if self._on_current:
            result |= cand_obj <= curr_obj

        if hasattr(self, "_n_last"):
            if not self._last_objectives:
                self._last_objectives.append(curr_obj)
                return cand_obj <= curr_obj

            result |= cand_obj <= self._last_objectives[0]
            self._last_objectives.append(curr_obj)

        return result
