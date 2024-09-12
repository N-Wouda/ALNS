import logging

import numpy as np

from alns.accept.update import update

logger = logging.getLogger(__name__)


class SimulatedAnnealing:
    R"""
    Simulated annealing, using an updating temperature.

    A candidate solution :math:`s^c` is compared against the current solution
    :math:`s`. The probability of accepting :math:`s^c` is given by

    .. math::

        \exp \left\{ \frac{f(s) - f(s^c)}{T} \right\},

    where :math:`T` is the current temperature, and :math:`f(\cdot)` gives the
    objective value of the passed-in solution. The current temperature
    :math:`T` is updated in each iteration using a step size :math:`\gamma`,
    as:

    .. math::

        T \gets \max \{ T_\text{end},~T - \gamma \}

    when ``method = 'linear'``, or

    .. math::

        T \gets \max \{ T_\text{end},~\gamma T \}

    when ``method = 'exponential'``. Initially, :math:`T` is set to
    :math:`T_\text{start}`.

    Parameters
    ----------
    start_temperature
        The initial temperature :math:`T_\text{start} > 0`.
    end_temperature
        The final temperature :math:`T_\text{end} > 0`.
    step
        The updating step size :math:`\gamma \ge 0`.
    method
        The updating method, one of {'linear', 'exponential'}. Default
        'exponential'.

    References
    ----------
    .. [1] Santini, A., Ropke, S. & Hvattum, L.M. A comparison of acceptance
           criteria for the adaptive large neighbourhood search metaheuristic.
           *Journal of Heuristics* (2018) 24 (5): 783-815.
    .. [2] Kirkpatrick, S., Gerlatt, C. D. Jr., and Vecchi, M. P. Optimization
           by Simulated Annealing. *IBM Research Report* RC 9355, 1982.
    """

    def __init__(
        self,
        start_temperature: float,
        end_temperature: float,
        step: float,
        method: str = "exponential",
    ):
        if start_temperature <= 0 or end_temperature <= 0 or step < 0:
            raise ValueError("Temperatures must be strictly positive.")

        if start_temperature < end_temperature:
            raise ValueError(
                "start_temperature < end_temperature not understood."
            )

        if method == "exponential" and step > 1:
            raise ValueError("Exponential updating cannot have step > 1.")

        self._start_temperature = start_temperature
        self._end_temperature = end_temperature
        self._step = step
        self._method = method

        self._temperature = start_temperature

    @property
    def start_temperature(self) -> float:
        return self._start_temperature

    @property
    def end_temperature(self) -> float:
        return self._end_temperature

    @property
    def step(self) -> float:
        return self._step

    @property
    def method(self) -> str:
        return self._method

    def __call__(self, rng, best, current, candidate):
        probability = np.exp(
            (current.objective() - candidate.objective()) / self._temperature
        )

        # We should not set a temperature that is lower than the end
        # temperature.
        self._temperature = max(
            self.end_temperature,
            update(self._temperature, self.step, self.method),
        )

        return probability >= rng.random()

    @classmethod
    def autofit(
        cls,
        init_obj: float,
        worse: float,
        accept_prob: float,
        num_iters: int,
        method: str = "exponential",
    ) -> "SimulatedAnnealing":
        """
        Returns an SA object with initial temperature such that there is a
        ``accept_prob`` chance of selecting a solution up to ``worse`` percent
        worse than the initial solution. The step parameter is then chosen such
        that the temperature reaches 1 in ``num_iters`` iterations.

        This procedure was originally proposed by Ropke and Pisinger (2006),
        and has seen some use since - i.a. Roozbeh et al. (2018).

        Parameters
        ----------
        init_obj
            The initial solution objective.
        worse
            Percentage (in (0, 1), exclusive) the candidate solution may be
            worse than initial solution for it to be accepted with probability
            ``accept_prob``.
        accept_prob
            Initial acceptance probability (in [0, 1]) for a solution at most
            ``worse`` worse than the initial solution.
        num_iters
            Number of iterations the ALNS algorithm will run.
        method
            The updating method, one of {'linear', 'exponential'}. Default
            'exponential'.

        Raises
        ------
        ValueError
            When the parameters do not meet requirements.

        Returns
        -------
        SimulatedAnnealing
            An autofitted SimulatedAnnealing acceptance criterion.

        References
        ----------
        .. [1] Ropke, Stefan, and David Pisinger. 2006. "An Adaptive Large
               Neighborhood Search Heuristic for the Pickup and Delivery
               Problem with Time Windows." *Transportation Science* 40 (4): 455
               - 472.
        .. [2] Roozbeh et al. 2018. "An Adaptive Large Neighbourhood Search for
               asset protection during escaped wildfires."
               *Computers & Operations Research* 97: 125 - 134.
        """
        if not (0 <= worse <= 1):
            raise ValueError("worse outside [0, 1] not understood.")

        if not (0 < accept_prob < 1):
            raise ValueError("accept_prob outside (0, 1) not understood.")

        if num_iters <= 0:
            raise ValueError("Non-positive num_iters not understood.")

        if method not in ["linear", "exponential"]:
            raise ValueError("Method must be one of ['linear', 'exponential']")

        start_temp = -worse * init_obj / np.log(accept_prob)

        if method == "linear":
            step = (start_temp - 1) / num_iters
        else:
            step = (1 / start_temp) ** (1 / num_iters)

        logger.info(
            f"Autofit {method} SA: start_temp {start_temp:.2f}, "
            f"step {step:.2f}."
        )

        return cls(start_temp, 1, step, method=method)
