import numpy as np


class AdaptiveThreshold:
    """
    The Adaptive Threshold (AT) criterion accepts solutions if the candidate
    solution has a value lower than an adaptive threshold. The adaptive threshold is
    computed as

    ''adaptive_threshold =  best_solution + heta_parameter * (average_solution - best_solution)''

    where
    ``best_solution`` is the best solution received so far,
    ``average_solution`` is the average of the last ``gamma_parameter`` solutions received, and
    ``heta_parameter`` is a parameter between 0 and 1, the greater the value of ``heta_parameter``,
    the more likely it is that a solution will be accepted.

    Each time a new solution is received, the threshold is updated. The average solution and best solution
    are taken by the last "gamma_parameter" solutions received. If the number of solutions received is less than
    "gamma_parameter" then the threshold is updated with the average of all the solutions received so far.

    The implementation is based on the description of AT in [1].

    Parameters
    ----------
    heta
        Parameter used to update the threshold ,the greater the value of ``heta_parameter``,
        the more likely it is that a solution will be accepted.
    gamma
        Parameter used to update the threshold, the number of solutions received to compute the average & best solution.

    References ---------- .. [1] Vinícius R. Máximo, Mariá C.V. Nascimento: A hybrid adaptive iterated local search
    with diversification control to the capacitated vehicle routing problem, European Journal of Operational
    Research, Volume 294, Issue 3,2021, Pages 1108-1119, ISSN 0377-2217.

    """

    def __init__(self, heta: float, gamma: float):
        if (heta > 1 or heta < 0) or (0 >= gamma):
            raise ValueError(
                "heta must be between 0 and 1, and gamma must be greater than 0."
            )

        self._heta = heta
        self._gamma = gamma
        self._solutions_received = np.array([])

    @property
    def heta(self):
        return self.heta

    @property
    def gamma(self):
        return self.gamma

    @property
    def solutions_received(self):
        return self.solutions_received

    def __call__(self, candidate):
        self._solutions_received = np.append(
            self._solutions_received, candidate.objective()
        )
        if len(self._solutions_received) <= self._gamma:
            best_solution = min(self._solutions_received)
            avg_solution = (
                sum(self._solutions_received) / self._solutions_received.size
            )
        else:
            self._solutions_received = np.delete(self._solutions_received, 0)
            best_solution = min(self._solutions_received)
            avg_solution = (
                sum(self._solutions_received) / self._solutions_received.size
            )
        res = candidate.objective() <= best_solution + self._heta * (
            avg_solution - best_solution
        )
        return res
