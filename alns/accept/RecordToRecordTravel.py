import logging

from alns.accept.update import update

logger = logging.getLogger(__name__)


class RecordToRecordTravel:
    """
    The Record-to-Record Travel (RRT) criterion accepts a candidate solution
    if the absolute gap between the candidate and the best or current solution
    is smaller than a threshold. The threshold :math:`T` is updated in each
    iteration using a step size :math:`\\gamma`, as:

    .. math::

        T \\gets \\max \\{ T_\\text{end},~T - \\gamma \\}

    when ``method = 'linear'``, or

    .. math::

        T \\gets \\max \\{ T_\\text{end},~\\gamma T \\}

    when ``method = 'exponential'``. Initially, :math:`T` is set to
    :math:`T_\\text{start}`.

    Parameters
    ----------
    start_threshold
        The initial threshold :math:`T_\\text{start} \\ge 0`.
    end_threshold
        The final threshold :math:`T_\\text{end} \\ge 0`.
    step
        The updating step size :math:`\\gamma \\ge 0`.
    method
        The updating method, one of {'linear', 'exponential'}. Default
        'linear'.
    cmp_best
        This parameter determines whether we use default RRT (True), or
        threshold accepting (False). By default, `cmp_best` is True, in which
        case RRT checks whether the difference between the candidate and best
        solution is below the threshold [2]. If `cmp_best` is False, RRT takes
        the difference between the candidate and current solution instead. This
        yields the behaviour of threshold accepting (TA), see [3] for details.

    References
    ----------
    .. [1] Santini, A., Ropke, S. & Hvattum, L.M. A comparison of acceptance
           criteria for the adaptive large neighbourhood search metaheuristic.
           *Journal of Heuristics* (2018) 24 (5): 783-815.
    .. [2] Dueck, G. New optimization heuristics: The great deluge algorithm
           and the record-to-record travel. *Journal of Computational Physics*
           (1993) 104 (1): 86-92.
    .. [3] Dueck, G., Scheuer, T. Threshold accepting: A general purpose
         optimization algorithm appearing superior to simulated annealing.
         *Journal of Computational Physics* (1990) 90 (1): 161-175.
    """

    def __init__(
        self,
        start_threshold: float,
        end_threshold: float,
        step: float,
        method: str = "linear",
        cmp_best: bool = True,
    ):
        if start_threshold < 0 or end_threshold < 0 or step < 0:
            raise ValueError("Thresholds and step must be non-negative.")

        if start_threshold < end_threshold:
            raise ValueError("start_threshold < end_threshold not understood.")

        if method == "exponential" and step > 1:
            raise ValueError("Exponential updating cannot have step > 1.")

        if method not in ["linear", "exponential"]:
            raise ValueError("Method must be one of ['linear', 'exponential']")

        self._start_threshold = start_threshold
        self._end_threshold = end_threshold
        self._step = step
        self._method = method
        self._cmp_best = cmp_best

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

    def __call__(self, rng, best, current, candidate):
        # From [2] p. 87 (RRT; best), and [3] p. 162 (TA; current).
        baseline = best if self._cmp_best else current
        res = candidate.objective() - baseline.objective() <= self._threshold

        self._threshold = max(
            self.end_threshold, update(self._threshold, self.step, self.method)
        )

        return res

    @classmethod
    def autofit(
        cls,
        init_obj: float,
        start_gap: float,
        end_gap: float,
        num_iters: int,
        method: str = "linear",
    ):
        """
        Returns an RRT object such that the start threshold is set at
        ``start_gap`` percent of the initial objective ``init_obj``
        and the end threshold is set at ``end_gap`` percent of the
        initial objective. The step parameter is then chosen such that
        the end threshold is reached in ``num_iters`` iterations using
        the passed-in update method.

        Parameters
        ----------
        init_obj
            The initial solution objective
        start_gap
            Percentage gap of the initial objective used for deriving
            the start threshold.
        end_gap
            Percentage gap of the initial objective used for deriving
            the end threshold.
        num_iters
            The number of iterations that the ALNS algorithm will run.
        method
            The updating method, one of {'linear', 'exponential'}. Default
            'linear'.

        Raises
        ------
        ValueError
            When the parameters do not meet requirements.

        Returns
        -------
        RecordToRecordTravel
            An autofitted RecordToRecordTravel acceptance criterion.
        """
        if not (0 <= end_gap <= start_gap):
            raise ValueError("Must have 0 <= end_gap <= start_gap")

        if num_iters <= 0:
            raise ValueError("Non-positive num_iters not understood.")

        if method not in ["linear", "exponential"]:
            raise ValueError("Method must be one of ['linear', 'exponential']")

        start_threshold = start_gap * init_obj
        end_threshold = end_gap * init_obj

        if method == "linear":
            step = (start_threshold - end_threshold) / num_iters
        else:
            step = (end_threshold / start_threshold) ** (1 / num_iters)

        logger.info(
            f"Autofit {method} RRT: start_threshold {start_threshold:.2f}, "
            f"end_threshold {end_threshold:.2f}, step {step:.2f}."
        )

        return cls(start_threshold, end_threshold, step, method=method)
