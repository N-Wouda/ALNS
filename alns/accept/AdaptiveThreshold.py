from statistics import mean
from typing import List


class AdaptiveThreshold:
    """
    The Adaptive Threshold (AT) criterion accepts solutions
    if the candidate solution has a value lower than an
    adaptive threshold. The adaptive threshold is computed as:

    ''adaptive_threshold =  best_solution +
    eta_parameter * (average_solution - best_solution)''

    where
    ``best_solution`` is the best solution received so far,
    ``average_solution`` is the average of the last
    ``gamma_parameter`` solutions received, and
    ``eta_parameter`` is a parameter between 0 and 1,
    the greater the value of
    ``eta_parameter``, the more likely it is that a solution
    will be accepted.

    Each time a new solution is received,
    the threshold is updated. The average solution
    and best solution are taken by the last "gamma_parameter"
    solutions received. If the number of solutions received
    is less than"gamma_parameter" then the threshold
    is updated with the average of all the solutions
    received so far.

    The implementation is based on the description of AT in [1].

    Parameters
    ----------
    eta: float
        Used to update/tune the threshold,
        the greater the value of ``eta_parameter``,
        the more likely it is that a solution will be accepted.
    gamma: int
        Used to update the threshold, the number of solutions
        received to compute the average & best solution.

    References
    ----------
    .. [1] Vinícius R. Máximo, Mariá C.V. Nascimento 2021.
           "A hybrid adaptive iterated local search with
           diversification control to the capacitated
           vehicle routing problem."
           *European Journal of Operational Research*
           294 (3): 1108 - 1119.
    """

    def __init__(self, eta: float, gamma: int):
        if (eta > 1 or eta < 0) or (0 > gamma):
            raise ValueError(
                "eta must be between 0 and 1, "
                "and gamma must be greater than 0."
            )

        self._eta = eta
        self._gamma = gamma
        self._history: List[float] = []

    @property
    def eta(self) -> float:
        return self._eta

    @property
    def gamma(self) -> int:
        return self._gamma

    @property
    def history(self) -> List[float]:
        return self._history

    def __call__(self, rnd, best, current, candidate) -> bool:
        self._history.append(candidate.objective())
        if len(self._history) > self._gamma:
            self._history = self._history[1:]
        best_solution = min(self._history)
        avg_solution = mean(self._history)
        threshold = best_solution + self._eta * (avg_solution - best_solution)
        return candidate.objective() <= threshold
