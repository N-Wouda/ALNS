from collections import deque

from alns.accept.AcceptanceCriterion import AcceptanceCriterion


class LateAcceptanceHillClimbing(AcceptanceCriterion):
    """
    The Late Acceptance Hill Climbing (LAHC) criterion accepts a candidate
    solution when it is better than the current solution from a number of
    iterations ago.

    This implementation is based on the description of LAHC in [1].

    Parameters
    ----------
    lookback_period: int
        Nonnegative integer specifying which solution to compare against
        for late acceptance. In particular, LAHC compares against the
        then-current solution from `lookback_period` iterations ago.
        If set to 0, then LAHC reverts to regular hill climbing.
    greedy: bool
        If set, LAHC always accepts a candidate that is better than the
        current solution.
    better_history: bool
        If set, LAHC uses a history management strategy where current solutions
        are stored only if they improve the then-current solution from
        `lookback_period` iterations ago; otherwise the then-current solution
        is stored again.

    References
    ----------
    [1]: Burke, E. K., & Bykov, Y. The late acceptance hill-climbing heuristic.
         *European Journal of Operational Research* (2017), 258(1), 70-78.
    """

    def __init__(
        self,
        lookback_period: int,
        greedy: bool = False,
        better_history: bool = False,
    ):
        self._lookback_period = lookback_period
        self._greedy = greedy
        self._better_history = better_history

        if not isinstance(lookback_period, int) or lookback_period < 0:
            raise ValueError("lookback_period must be a non-negative integer.")

        self._history: deque = deque([], maxlen=lookback_period)

    @property
    def lookback_period(self):
        return self._lookback_period

    @property
    def greedy(self):
        return self._greedy

    @property
    def better_history(self):
        return self._better_history

    def __call__(self, rnd, best, curr, cand):
        if not self._history:
            self._history.append(curr.objective())
            return cand.objective() < curr.objective()

        res = cand.objective() < self._history[0]

        if not res and self._greedy:
            res = cand.objective() < curr.objective()

        if self._better_history:
            self._history.append(min(curr.objective(), self._history[0]))
        else:
            self._history.append(curr.objective())

        return res
