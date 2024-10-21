import math

from alns.accept.GreatDeluge import GreatDeluge


class NonLinearGreatDeluge(GreatDeluge):
    """
    The Non-Linear Great Deluge (NLGD) criterion accepts solutions if the
    candidate solution has a value lower than a threshold (originally called
    the water level [1]). The initial threshold is computed as

    ``threshold = alpha * initial.objective()``

    where ``initial`` is the initial solution passed-in to ALNS.

    The non-linear GD variant was proposed by [2]. It differs from GD by using
    a non-linear updating scheme; see the ``_compute_threshold`` method for
    details. Moreover, candidate solutions that improve the current solution
    are always accepted.

    The implementation is based on the description in [2].

    Parameters
    ----------
    alpha
        Factor used to compute the initial threshold. See [2] for details.
    beta
        Factor used to update the threshold. See [2] for details.
    gamma
        Factor used to update the threshold. See [2] for details.
    delta
        Factor used to update the threshold. See [2] for details.

    References
    ----------
    .. [1] Dueck, G. New optimization heuristics: The great deluge algorithm
           and the record-to-record travel. *Journal of Computational Physics*
           (1993) 104 (1): 86-92.
    .. [2] Landa-Silva, D., & Obit, J. H. Great deluge with non-linear decay
           rate for solving course timetabling problems. *4th international
           IEEE conference intelligent systems* (2008) Vol. 1: 8-11.
    .. [3] Santini, A., Ropke, S. & Hvattum, L.M. A comparison of acceptance
           criteria for the adaptive large neighbourhood search metaheuristic.
           *Journal of Heuristics* (2018) 24 (5): 783-815.
    """

    def __init__(self, alpha: float, beta: float, gamma: float, delta: float):
        super().__init__(alpha, beta)

        if gamma <= 0 or delta <= 0:
            raise ValueError("Gamma and delta must be non-negative.")

        self._gamma = gamma
        self._delta = delta

    @property
    def gamma(self):
        return self._gamma

    @property
    def delta(self):
        return self._delta

    def __call__(self, rng, best, current, candidate):
        if self._threshold is None:
            if best.objective() == 0:
                raise ValueError("Initial solution cannot have zero value.")

            self._threshold = self._alpha * best.objective()

        res = candidate.objective() < self._threshold

        if not res:
            # Accept if improving
            res = candidate.objective() < current.objective()

        self._threshold = self._compute_threshold(best, current, candidate)

        return res

    def _compute_threshold(self, best, curr, cand):
        """
        Returns the new threshold value.

        First, the relative gap between the candidate solution and threshold
        is computed. If this relative gap is less than ``beta``, then the
        threshold is linearly increased (involving the ``gamma`` parameter).
        Otherwise, the threshold is exponentially decreased (involving the
        ``delta`` parameter).
        """
        rel_gap = (self._threshold - cand.objective()) / self._threshold

        if rel_gap < self._beta:
            term1 = self.gamma * abs(cand.objective() - self._threshold)
            term2 = self._threshold
        else:
            term1 = self._threshold * math.exp(-self.delta * best.objective())
            term2 = best.objective()

        return term1 + term2
