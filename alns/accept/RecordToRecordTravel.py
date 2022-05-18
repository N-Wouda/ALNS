from alns.accept.AcceptanceCriterion import AcceptanceCriterion
from alns.accept.update import update


class RecordToRecordTravel(AcceptanceCriterion):
    """
    Record-to-record travel, using an updating threshold. The threshold is
    updated as,

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

    def __init__(
        self,
        start_threshold: float,
        end_threshold: float,
        step: float,
        method: str = "linear",
    ):
        if start_threshold < 0 or end_threshold < 0 or step < 0:
            raise ValueError("Thresholds must be positive.")

        if start_threshold < end_threshold:
            raise ValueError("start_threshold < end_threshold not understood.")

        if method == "exponential" and step > 1:
            raise ValueError("Exponential updating cannot have step > 1.")

        self._start_threshold = start_threshold
        self._end_threshold = end_threshold
        self._step = step
        self._method = method

        self._threshold = start_threshold

    @property
    def start_threshold(self) -> float:
        return self._start_threshold

    @property
    def end_threshold(self) -> float:
        return self._end_threshold

    @property
    def step(self) -> float:
        return self._step

    @property
    def method(self) -> str:
        return self._method

    def __call__(self, rnd, best, current, candidate):
        # This follows from the paper by Dueck and Scheueur (1990), p. 162.
        result = (candidate.objective() - best.objective()) <= self._threshold

        self._threshold = max(
            self.end_threshold, update(self._threshold, self.step, self.method)
        )

        return result
