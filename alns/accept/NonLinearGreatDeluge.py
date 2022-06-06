import math
from typing import Optional

from alns.accept.AcceptanceCriterion import AcceptanceCriterion


class NonLinearGreatDeluge(AcceptanceCriterion):
    """
    The Non Linear Great Deluge (NLGD) criterion accepts solutions if the
    candidate solution has value lower than a threshold (originally called the
    water level). The initial threshold is computed as

    ``threshold = alpha * initial.objective()``

    where `initial` is the initial solution passed-in to ALNS, inferred
    from the best solution at the first iteration.

    The non-linear GD variant was proposed by [3]. It differs from GD by using
    a non-linear updating scheme, see the `_update` method for more details.
    Moreover, candidate solutions that improve the current solution are always
    accepted.

    The implementation is based on the description in [1].

    Parameters
    ----------
    alpha
        Factor to compute the initial threshold
    beta
        Factor used for updating the threshold
    gamma
        Factoe used for updating the threshold
    delta
        Factor used for updating the threshold

    References
    ----------
    [1]: Santini, A., Ropke, S. & Hvattum, L.M. A comparison of acceptance
         criteria for the adaptive large neighbourhood search metaheuristic.
         *Journal of Heuristics* (2018) 24 (5): 783–815.
    [2]: Dueck, G. New optimization heuristics: The great deluge algorithm and
         the record-to-record travel. *Journal of Computational Physics* (1993)
         104 (1): 86-92.
    [3]: Landa-Silva, D., & Obit, J. H. Great deluge with non-linear decay rate
         for solving course timetabling problems. *4th international IEEE
         conference intelligent systems* (2008) Vol. 1: 8-11.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
    ):
        if alpha <= 1:
            raise ValueError("Alpha must be larger than 1.")

        if not (0 < beta < 1):
            raise ValueError("Beta must be in (0, 1).")

        if gamma <= 0 or delta <= 0:
            raise ValueError("Gamma and delta must be non-negative.")

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._threshold: Optional[float] = None

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma

    @property
    def delta(self):
        return self._delta

    def __call__(self, rnd, best, curr, cand):
        if best.objective() == 0:
            raise ValueError("Initial solution cannot have zero value.")

        if self._threshold is None:
            self._threshold = self._alpha * best.objective()

        res = cand.objective() < self._threshold

        if not res:
            res |= cand.objective() < curr.objective()  # Accept improving

        self._threshold = self._update(best, curr, cand)

        return res

    def _update(self, best, curr, cand):
        """
        Return the new threshold value.

        First, the relative gap between the candidate solution and threshold
        is computed. If this relative gap is less than ``beta``, then the
        threshold is linearly increased (involving the `gamma` parameter).
        Otherwise, the threshold is exponentially decreased (involcing the
        `delta` parameter).
        """

        rel_gap = (self._threshold - cand.objective()) / self._threshold

        if rel_gap < self._beta:
            res = self._threshold + self._gamma * abs(
                cand.objective() - self._threshold
            )
        else:
            res = (
                self._threshold * math.exp(-self._delta * best.objective())
                + best.objective()
            )

        return res
