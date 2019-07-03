import numpy as np


class Statistics:

    def __init__(self):
        self._objectives = []

    @property
    def objectives(self):
        return np.array(self._objectives)

    def collect_objective(self, objective):
        self._objectives.append(objective)
