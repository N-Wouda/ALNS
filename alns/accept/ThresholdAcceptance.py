from alns.accept.RecordToRecordTravel import RecordToRecordTravel
from alns.accept.update import update


class ThresholdAcceptance(RecordToRecordTravel):
    """
    The Threshold Acceptance (TA) criterion accepts a candidate solution if
    the absolute gap between the candidate and current solution is smaller
    than a threshold. The threshold is updated in each iteration as:

    ``threshold = max(end_threshold, threshold - step)`` (linear)

    ``threshold = max(end_threshold, step * threshold)`` (exponential)

    where the initial threshold is set to ``start_threshold``.

    Parameters
    ----------
    start_threshold
        The initial threshold.
    end_threshold
        The final threshold.
    step
        The updating step.
    method
        The updating method, one of {'linear', 'exponential'}. Default
        'linear'.

    References
    ----------
    [1]: Santini, A., Ropke, S. & Hvattum, L.M. A comparison of acceptance
         criteria for the adaptive large neighbourhood search metaheuristic.
         *Journal of Heuristics* (2018) 24 (5): 783–815.
    [2]: Dueck, G., Scheuer, T. Threshold accepting: A general purpose
         optimization algorithm appearing superior to simulated annealing.
         *Journal of Computational Physics* (1990) 90 (1): 161-175.
    """

    def __call__(self, rnd, best, current, candidate):
        # This follows from the paper by Dueck and Scheueur (1990), p. 162.
        diff = candidate.objective() - current.objective()
        res = diff <= self._threshold

        self._threshold = max(
            self.end_threshold, update(self._threshold, self.step, self.method)
        )

        return res
