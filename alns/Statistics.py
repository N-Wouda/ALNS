import numpy as np


class Statistics:

    def __init__(self):
        """
        Statistics object that stores some iteration results, which is
        optionally populated by the ALNS algorithm.
        """
        self._objectives = []

    @property
    def objectives(self):
        """
        Returns an array of previous objective values, tracking progress.
        """
        return np.array(self._objectives)

    def collect_objective(self, objective):
        """
        Collects an objective value.

        Parameters
        ----------
        objective : float
            The objective value to be collected.
        """
        self._objectives.append(objective)
