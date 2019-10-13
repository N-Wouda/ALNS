from collections import defaultdict

import numpy as np


class Statistics:

    def __init__(self):
        """
        Statistics object that stores some iteration results, which is
        optionally populated by the ALNS algorithm.
        """
        self._objectives = []

        # TODO: is this as operator idx or name? If name, add to ALNS field.
        self._destroy_operator_counts = defaultdict(lambda: [0, 0, 0, 0])
        self._repair_operator_counts = defaultdict(lambda: [0, 0, 0, 0])

    @property
    def objectives(self):
        """
        Returns an array of previous objective values, tracking progress.
        """
        return np.array(self._objectives)

    @property
    def destroy_operator_counts(self):
        """
        TODO
        """
        return self._destroy_operator_counts

    @property
    def repair_operator_counts(self):
        """
        TODO
        """
        return self._repair_operator_counts

    def collect_objective(self, objective):
        """
        Collects an objective value.

        Parameters
        ----------
        objective : float
            The objective value to be collected.
        """
        self._objectives.append(objective)

    def collect_repair_operator_count(self, operator, weight):
        """
        TODO
        """
        pass

    def collect_destroy_operator_count(self, operator, weight):
        """
        TODO
        """
        pass
