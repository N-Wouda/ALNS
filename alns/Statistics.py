from collections import defaultdict
from typing import DefaultDict, List

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
    def objectives(self) -> np.ndarray:
        """
        Returns an array of previous objective values, tracking progress.
        """
        return np.array(self._objectives)

    @property
    def destroy_operator_counts(self) -> DefaultDict[str, List[float]]:
        """
        Returns the destroy operator counts, as a dictionary of operator names
        to lists of counts. Such a list consists of four elements, one for
        each possible outcome, and counts the number of times that the
        application of that operator resulted in such an outcome.

        Returns
        -------
        Destroy operator counts.
        """
        return self._destroy_operator_counts

    @property
    def repair_operator_counts(self) -> DefaultDict[str, List[float]]:
        """
        Returns the repair operator counts, as a dictionary of operator names
        to lists of counts. Such a list consists of four elements, one for
        each possible outcome, and counts the number of times that the
        application of that operator resulted in such an outcome.

        Returns
        -------
        Repair operator counts.
        """
        return self._repair_operator_counts

    def collect_objective(self, objective: float):
        """
        Collects an objective value.

        Parameters
        ----------
        objective
            The objective value to be collected.
        """
        self._objectives.append(objective)

    def collect_destroy_operator(self, operator_name: str, s_idx: int):
        """
        Collects a score (index) for a used destroy operator. This maintains
        count of the number of times this operator was used, and what result
        came from its use.

        Parameters
        ----------
        operator_name
            Operator name. This was set when the operator was passed to the
            ALNS instance.
        s_idx
            Score indices used for the various iteration outcomes.
        """
        self._destroy_operator_counts[operator_name][s_idx] += 1

    def collect_repair_operator(self, operator_name: str, s_idx: int):
        """
        Collects a score (index) for a used repair operator. This maintains
        count of the number of times this operator was used, and what result
        came from its use.

        Parameters
        ----------
        operator_name
            Operator name. This was set when the operator was passed to the
            ALNS instance.
        s_idx
            Score indices used for the various iteration outcomes.
        """
        self._repair_operator_counts[operator_name][s_idx] += 1
