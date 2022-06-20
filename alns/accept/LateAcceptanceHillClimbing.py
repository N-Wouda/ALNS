from collections import deque

from alns.accept.AcceptanceCriterion import AcceptanceCriterion


class LateAcceptanceHillClimbing(AcceptanceCriterion):
    """
    The Late Acceptance Hill Climbing (LAHC) criterion accepts a candidate
    solution when it is better than the current solution from a number of
    iterations before.

    This implementation is based on the description of LAHC in [1].

    Parameters
    ----------
    history_length: int
        Non-negative integer specifying the maximum number of previous solutions
        to be stored. Default: 0 (i.e., no objectives stored).
    greedy: bool
        Bool indicating whether or not to accept candidate solutions if they are
        better than the current solution. Default: False.
    collect_better: bool
        Bool indicating whether or not to only collect current solutions that
        are better than the previous current. Default: False

    References
    ----------
    [1]: Burke, E. K., & Bykov, Y. The late acceptance hill-climbing heuristic.
         *European Journal of Operational Research* (2017), 258(1), 70-78.
    """

    def __init__(
        self,
        history_length: int = 0,
        greedy: bool = False,
        collect_better: bool = False,
    ):
        self._history_length = history_length
        self._greedy = greedy
        self._collect_better = collect_better

        if not isinstance(history_length, int) or history_length < 0:
            raise ValueError("history_length must be a non-negative integer.")

        self._objectives: deque = deque([], maxlen=history_length)

    @property
    def history_length(self):
        return self._history_length

    @property
    def greedy(self):
        return self._greedy

    @property
    def collect_better(self):
        return self._collect_better

    def __call__(self, rnd, best, curr, cand):
        cand_obj = cand.objective()
        curr_obj = curr.objective()

        if self._objectives and self._collect_better:
            self._objectives.append(min(curr_obj, self._objectives[0]))
        else:
            self._objectives.append(curr_obj)

        res = cand_obj < self._objectives[0]

        if not res and self._greedy:
            res = cand_obj < curr_obj  # Accept if improving

        return res
