from typing import Optional

from alns.accept.AcceptanceCriterion import AcceptanceCriterion


class GreatDeluge(AcceptanceCriterion):
    """
    The Great Deluge (GD) criterion accepts solutions if the candidate solution
    has value lower than a threshold (originally called the water level). The
    initial threshold is computed as

    ``threshold = alpha * initial.objective()``

    where `initial` is the initial solution passed-in to ALNS, inferred
    from the best solution at the first iteration. The threshold is updated in
    each iteration as

    ``threshold = threshold - beta * (threshold - candidate.objective()``

    The implementation is based on the description in [1].

    Parameters
    ----------
    alpha
        Factor to compute the initial threshold
    beta
        Factor used for updating the threshold

    References
    ----------
    [1]: Santini, A., Ropke, S. & Hvattum, L.M. A comparison of acceptance
         criteria for the adaptive large neighbourhood search metaheuristic.
         *Journal of Heuristics* (2018) 24 (5): 783–815.
    [2]: Dueck, G. New optimization heuristics: The great deluge algorithm and
         the record-to-record travel. *Journal of Computational Physics* (1993)
         104 (1): 86-92.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
    ):
        if alpha <= 1:
            raise ValueError("Alpha must be larger than 1.")

        if not (0 < beta < 1):
            raise ValueError("Beta must be in (0, 1).")

        self._alpha = alpha
        self._beta = beta
        self._threshold: Optional[float] = None

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def __call__(self, rnd, best, current, candidate):
        if self._threshold is None:
            self._threshold = self._alpha * best.objective()

        res = candidate.objective() < self._threshold

        self._threshold = self._update(best, current, candidate)

        return res

    def _update(self, best, current, candidate):
        change = self._beta * (candidate.objective() - self._threshold)
        return self._threshold + change
