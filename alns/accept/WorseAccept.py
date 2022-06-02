from alns.accept.AcceptanceCriterion import AcceptanceCriterion
from alns.accept.update import update


class WorseAccept(AcceptanceCriterion):
    """
    The worse accept criterion accepts a candidate solution if it improves over
    the current one, or - regardless of the cost - with a given probability.

    The probability is updated over time.

    Parameters
    ----------
    start_prob
        The initial probability.
    end_prob
        The final probability.
    step:
        The updating step.
    method
        The updating method, one of {'linear', 'exponential'}. Default
        'linear'.

    """

    def __init__(
        self,
        start_prob: float,
        end_prob: float,
        step: float,
        method: str = "linear",
    ):
        if not 0 <= start_prob <= 1 or not 0 <= end_prob <= 1:
            raise ValueError("Probabilities must be in [0, 1].")

        if start_prob < end_prob:
            raise ValueError(
                "Start probability < end probability not understood."
            )

        if step < 0:
            raise ValueError("Step cannot be negative.")

        if method == "exponential" and step > 1:
            raise ValueError("Exponential updating cannot have step > 1.")

        self._start_prob = start_prob
        self._end_prob = end_prob
        self._step = step
        self._method = method

        self._prob = start_prob

    @property
    def start_prob(self) -> float:
        return self._start_prob

    @property
    def end_prob(self) -> float:
        return self._end_prob

    @property
    def step(self) -> float:
        return self._step

    @property
    def method(self) -> str:
        return self._method

    def __call__(self, rnd, best, current, candidate):
        result = (
            candidate.objective() < current.objective()
            or rnd.random() < self._prob
        )

        self._prob = max(
            self.end_prob, update(self._prob, self.step, self.method)
        )

        return result
