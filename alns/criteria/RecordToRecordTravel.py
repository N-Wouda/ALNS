from .AcceptanceCriterion import AcceptanceCriterion


class RecordToRecordTravel(AcceptanceCriterion):
    """
    Linear record-to-record travel, using an updating threshold. The threshold
    is updated as,

    ``threshold = max(end_threshold, threshold - step)``

    where the initial threshold is set to ``start_threshold``.

    Parameters
    ----------
    start_threshold : float
        The initial threshold.
    end_threshold : float
        The final threshold.
    step : float
        The updating step.

    References
    ----------
    - Santini, A., Ropke, S. & Hvattum, L.M. A comparison of acceptance
      criteria for the adaptive large neighbourhood search metaheuristic.
      *Journal of Heuristics* (2018) 24 (5): 783–815.
    - Dueck, G., Scheuer, T. Threshold accepting: A general purpose
      optimization algorithm appearing superior to simulated annealing.
      *Journal of Computational Physics* (1990) 90 (1): 161-175.
    """

    def __init__(self, start_threshold, end_threshold, step):
        if start_threshold < 0 or end_threshold < 0 or step < 0:
            raise ValueError("Thresholds must be positive.")

        if start_threshold < end_threshold:
            raise ValueError("Start threshold must be bigger than end "
                             "threshold.")

        self._start_threshold = start_threshold
        self._end_threshold = end_threshold
        self._step = step

        self._threshold = start_threshold

    @property
    def start_threshold(self):
        return self._start_threshold

    @property
    def end_threshold(self):
        return self._end_threshold

    @property
    def step(self):
        return self._step

    def accept(self, best, current, candidate):
        # This follows from the paper by Dueck and Scheueur (1990), p. 162.
        result = (candidate.objective() - best.objective()) <= self._threshold

        # We should not set a threshold that is lower than the end threshold.
        self._threshold = max(self.end_threshold, self._threshold - self.step)

        return result
