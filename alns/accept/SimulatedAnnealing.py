import logging

import numpy as np

from alns.accept.AcceptanceCriterion import AcceptanceCriterion
from alns.accept.update import update

logger = logging.getLogger(__name__)


class SimulatedAnnealing(AcceptanceCriterion):
    """
    Simulated annealing, using an updating temperature. The temperature is
    updated as,

    ``temperature = max(end_temperature, temperature - step)`` (linear)

    ``temperature = max(end_temperature, step * temperature)`` (exponential)

    where the initial temperature is set to ``start_temperature``.

    Parameters
    ----------
    start_temperature
        The initial temperature.
    end_temperature
        The final temperature.
    step
        The updating step.
    method
        The updating method, one of {'linear', 'exponential'}. Default
        'exponential'.

    References
    ----------
    [1]: Santini, A., Ropke, S. & Hvattum, L.M. A comparison of acceptance
         criteria for the adaptive large neighbourhood search metaheuristic.
         *Journal of Heuristics* (2018) 24 (5): 783–815.
    [2]: Kirkpatrick, S., Gerlatt, C. D. Jr., and Vecchi, M. P. Optimization
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

    def __call__(self, rnd, best, current, candidate):
        probability = np.exp(
            (current.objective() - candidate.objective()) / self._temperature
        )

        # We should not set a temperature that is lower than the end
        # temperature.
        self._temperature = max(
            self.end_temperature,
            update(self._temperature, self.step, self.method),
        )

        # TODO deprecate RandomState in favour of Generator - which uses
        #  random(), rather than random_sample().
        try:
            return probability >= rnd.random()
        except AttributeError:
            return probability >= rnd.random_sample()

    @classmethod
    def autofit(
        cls, init_obj: float, worse: float, accept_prob: float, num_iters: int
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
            Percentage (between 0 and 1) the candidate solution may be worse
            than initial solution for it to be accepted with probability
            ``accept_prob``.
        accept_prob
            Initial acceptance probability for a solution at most ``worse``
            worse than the initial solution.
        num_iters
            Number of iterations the ALNS algorithm will run.

        Raises
        ------
        ValueError
            When ``worse`` not in [0, 1] or when ``accept_prob``not in (0, 1).

        Returns
        -------
        An autofitted SimulatedAnnealing acceptance criterion.

        References
        ----------
        [1]: Ropke, Stefan, and David Pisinger. 2006. "An Adaptive Large
             Neighborhood Search Heuristic for the Pickup and Delivery Problem
             with Time Windows." _Transportation Science_ 40 (4): 455 - 472.
        [2]: Roozbeh et al. 2018. "An Adaptive Large Neighbourhood Search for
             asset protection during escaped wildfires." _Computers & Operations
             Research_ 97: 125 - 134.
        """
        if not (0 <= worse <= 1):
            raise ValueError("worse outside [0, 1] not understood.")

        if not (0 < accept_prob < 1):
            raise ValueError("accept_prob outside (0, 1) not understood.")

        if num_iters < 0:
            raise ValueError("Negative number of iterations not understood.")

        start_temp = -worse * init_obj / np.log(accept_prob)
        step = (1 / start_temp) ** (1 / num_iters)

        logger.info(f"Autofit start_temp {start_temp:.2f}, step {step:.2f}.")

        return cls(start_temp, 1, step, method="exponential")
