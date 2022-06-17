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
    n_iterations: int
        Non-negative integer specifying the number of iterations of previous
        current solution.
        last objectives to store.
        Default: 0 (no objectives stored).
    on_improve: bool
        Bool indicating whether or not to accept candidate solutions if they are
        better than the current solution. Default: False.
    only_better: bool
        Bool indicating whether or not to only store current solutions that are
        better than the previous current. Default: False

    References
    ----------
    [1]: Burke, E. K., & Bykov, Y. The late acceptance hill-climbing heuristic.
         *European Journal of Operational Research* (2017), 258(1), 70-78.
    """

    def __init__(
        self,
        n_iterations: int = 0,
        on_improve: bool = False,
        only_better: bool = False,
    ):
        self._n_iterations = n_iterations
        self._on_improve = on_improve
        self._only_better = only_better

        if not isinstance(n_iterations, int) or n_iterations < 0:
            raise ValueError("n_iterations must be a non-negative integer.")

        self._objectives: deque = deque([], maxlen=n_iterations)

    @property
    def n_iterations(self):
        return self._n_iterations

    @property
    def on_improve(self):
        return self._on_improve

    @property
    def only_better(self):
        return self._only_better

    def __call__(self, rnd, best, curr, cand):
        cand_obj = cand.objective()
        curr_obj = curr.objective()

        self._objectives.append(self._update(rnd, best, curr, cand))

        res = cand_obj < self._objectives[0]

        if not res and self._on_improve:
            res = cand_obj < curr_obj  # Accept if improving

        return res

    def _update(self, rnd, best, curr, cand):
        curr_obj = curr.objective()

        if not self._objectives or not self.only_better:
            result = curr_obj

        else:
            if curr_obj < self._objectives[0]:
                result = curr_obj
            else:
                result = self._objectives[0]

        return result
