from alns.accept.update import update


class RandomAccept:
    """
    The Random Accept criterion accepts a candidate solution if it improves
    over the current one, or with a given probability :math:`P` regardless of
    the cost. :math:`P` is updated in each iteration as:

    .. math::

        P \\gets \\max \\{ P_\\text{end},~P - \\gamma \\}

    when ``method = 'linear'``, or

    .. math::

        P \\gets \\max \\{ P_\\text{end},~\\gamma P \\}

    when ``method = 'exponential'``. Initially, :math:`P` is set to
    :math:`P_\\text{start}`.

    Parameters
    ----------
    start_prob
        The initial probability :math:`P_\\text{start} \\in [0, 1]`.
    end_prob
        The final probability :math:`P_\\text{end} \\in [0, 1]`.
    step:
        The updating step :math:`\\gamma \\ge 0`.
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

    def __call__(self, rng, best, current, candidate):
        # Always accept better
        res = candidate.objective() < current.objective()

        if not res:  # maybe accept worse
            res = rng.random() < self._prob

        self._prob = max(
            self.end_prob, update(self._prob, self.step, self.method)
        )

        return res
