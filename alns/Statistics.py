from collections import defaultdict

import numpy as np


class Statistics:

    def __init__(self):
        """
        Statistics object that stores some iteration results, which is
        optionally populated by the ALNS algorithm.
        """
        self._objectives = []

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
        Returns the destroy operator counts, as a dictionary of operator names
        to lists of counts. Such a list consists of four elements, one for
        each value in `WeightIndex`, and counts the number of times that the
        application of that operator resulted in such an outcome.

        Returns
        -------
        defaultdict
            Destroy operator counts.
        """
        return self._destroy_operator_counts

    @property
    def repair_operator_counts(self):
        """
        Returns the repair operator counts, as a dictionary of operator names
        to lists of counts. Such a list consists of four elements, one for
        each value in `WeightIndex`, and counts the number of times that the
        application of that operator resulted in such an outcome.

        Returns
        -------
        defaultdict
            Repair operator counts.
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

    def collect_destroy_operator(self, operator_name, weight_idx):
        """
        Collects a weight (index) for a used destroy operator. This maintains
        count of the number of times this operator was used, and what result
        came from its use.

        Parameters
        ----------
        operator_name : str
            Operator name. This was set when the operator was passed to the
            ALNS instance.
        weight_idx : int
            Weight indices used for the various iteration outcomes. See also
            the `WeightIndex` enum.
        """
        self._destroy_operator_counts[operator_name][weight_idx] += 1

    def collect_repair_operator(self, operator_name, weight_idx):
        """
        Collects a weight (index) for a used repair operator. This maintains
        count of the number of times this operator was used, and what result
        came from its use.

        Parameters
        ----------
        operator_name : str
            Operator name. This was set when the operator was passed to the
            ALNS instance.
        weight_idx : int
            Weight indices used for the various iteration outcomes. See also
            the `WeightIndex` enum.
        """
        self._repair_operator_counts[operator_name][weight_idx] += 1
