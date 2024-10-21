from typing import Optional


class GreatDeluge:
    """
    The Great Deluge (GD) criterion accepts solutions if the candidate solution
    has a value lower than a threshold (originally called the water level [1]).
    The initial threshold is computed as

    ``threshold = alpha * initial.objective()``

    where ``initial`` is the initial solution passed-in to ALNS.

    The threshold is updated in each iteration as

    ``threshold = threshold - beta * (threshold - candidate.objective())``

    The implementation is based on the description of GD in [2].

    Parameters
    ----------
    alpha
        Factor used to compute the initial threshold
    beta
        Factor used to update the threshold

    References
    ----------
    .. [1] Dueck, G. New optimization heuristics: The great deluge algorithm
           and the record-to-record travel. *Journal of Computational Physics*
           (1993) 104 (1): 86-92.
    .. [2] Santini, A., Ropke, S. & Hvattum, L.M. A comparison of acceptance
           criteria for the adaptive large neighbourhood search metaheuristic.
           *Journal of Heuristics* (2018) 24 (5): 783-815.
    """

    def __init__(self, alpha: float, beta: float):
        if alpha <= 1 or not (0 < beta < 1):
            raise ValueError("alpha must be > 1 and beta must be in (0, 1).")

        self._alpha = alpha
        self._beta = beta
        self._threshold: Optional[float] = None

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def __call__(self, rng, best, current, candidate):
        if self._threshold is None:
            self._threshold = self.alpha * best.objective()

        diff = self._threshold - candidate.objective()
        res = diff > 0

        self._threshold -= self.beta * diff

        return res
