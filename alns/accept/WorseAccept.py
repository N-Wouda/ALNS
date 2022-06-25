from alns.accept.AcceptanceCriterion import AcceptanceCriterion
from alns.accept.update import update


class WorseAccept(AcceptanceCriterion):
    """
    The Worse Accept criterion accepts a candidate solution 1) if it improves
    over the current one or 2) with a given probability regardless of the cost.
    The probability is updated in each iteration as:

    ``prob = max(end_prob, prob - step)`` (linear)

    ``prob = max(end_prob, step * prob)`` (exponential)

    where the initial probability is set to ``start_prob``.

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
        if not 0 <= end_prob <= start_prob <= 1:
            raise ValueError("Must have 0 <= start_prob <= end_prob <= 1")

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
        # Always accept better
        res = candidate.objective() < current.objective()

        if not res:  # maybe accept worse
            res = rnd.random() < self._prob

        self._prob = max(
            self.end_prob, update(self._prob, self.step, self.method)
        )

        return res
