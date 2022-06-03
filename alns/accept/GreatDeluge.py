import math
from typing import Optional

from alns.accept.AcceptanceCriterion import AcceptanceCriterion


class GreatDeluge(AcceptanceCriterion):
    """
    The Great Deluge (GD) criterion accepts solutions if the candidate solution
    has value smaller than an absolute threshold (originally called the water
    level). The implementation is based on the description in [1].

    In the GD criterion, the initial threshold is computed as

    ``threshold = alpha * initial.objective()``

    where ``initial`` is the initial solution passed-in to ALNS, inferred
    from the best solution at the first iteration.

    There are two variants of the GD criterion: 1) linear and 2) non-linear.
    The main difference between the variants is the threshold updating scheme,
    see the `_update` method for details. Moreover, in the non-linear variant,
    candidate solutions that improve the current solution are always accepted.

    Parameters
    ----------
    alpha
        The factor to compute the initial threshold
    beta
        A factor used for updating the threshold
    gamma
        A factor used for updating the threshold (only in non-linear GD)
    delta
        A factor used for updating the threshold (only in non-linear GD)
    method
        The updating method, one of {'linear', 'non-linear'}. Default 'linear'.

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
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
        method: str = "linear",
    ):
        if alpha <= 1:
            raise ValueError("Alpha must be larger than 1.")

        if beta <= 0 or beta >= 1:
            raise ValueError("Beta must be in (0, 1).")

        if (
            gamma is None or delta is None or gamma <= 0 or delta <= 0
        ) and method == "non-linear":
            raise ValueError(
                "Gamma and delta must non-negative if the non-linear method is selected."
            )

        if method not in ["linear", "non-linear"]:
            raise ValueError(f"Method {method} not understrood.")

        self._alpha = alpha
        self._beta = beta
        self._method = method
        self._threshold: Optional[float] = None

        if self._method == "non-linear":
            self._gamma: Optional[float] = gamma
            self._delta: Optional[float] = delta

    def __call__(self, rnd, best, current, candidate):
        if self._threshold is None:
            self._threshold = self._alpha * best.objective()

            if self._method == "non-linear" and self._threshold == 0:
                raise ValueError(
                    "Initial solution cannot have zero objective value in the non-linear method."
                )

        result = candidate.objective() < self._threshold

        if self._method == "non-linear":
            result |= candidate.objective() < current.objective()

        self._threshold = self._update(best, current, candidate)

        return result

    def _update(self, best, current, candidate):
        """
        In the linear GD variant, the threshold is updated in each iteration as

        ``threshold = threshold - beta * (threshold - candidate.objective())``

        In the non-linear variant, the updating method is more involved.
        - First, the relative gap between the candidate solution and threshold
          is computed.
          - If this relative gap is less than `beta` parameter, meaning
            that 1) the candidate value was higher than the threshold or 2)
            improving less than beta relative to the threshold, then the
            threshold is linearly increased (using the `gamma` parameter).
          - Otherwise, the threshold is exponentially decreased (using the
            `delta` parameter).
        """

        if self._method == "linear":
            result = self._threshold - self._beta * (
                self._threshold - candidate.objective()
            )

        elif self._method == "non-linear":
            rel_gap = (
                self._threshold - candidate.objective()
            ) / self._threshold

            if rel_gap < self._beta:
                # Linearly increase threshold
                result = self._threshold + self._gamma * abs(
                    candidate.objective() - self._threshold
                )
            else:
                # Exponentially decrease the threshold
                result = (
                    self._threshold * math.exp(-self._delta * best.objective())
                    + best.objective()
                )

        return result
