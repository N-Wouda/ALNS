import numpy as np

from .AcceptanceCriterion import AcceptanceCriterion


class SimulatedAnnealing(AcceptanceCriterion):

    def __init__(self, start_temperature, end_temperature, step):
        """
        Linear simulated annealing, using an updating temperature. The
        temperature is updated as,

        ``temperature = max(end_temperature, temperature - step)``

        where the initial temperature is set to ``start_temperature``.

        Parameters
        ----------
        start_temperature : float
            The initial temperature.
        end_temperature : float
            The final temperature.
        step : float
            The updating step.

        References
        ----------
        - Santini, A., Ropke, S. & Hvattum, L.M. A comparison of acceptance
          criteria for the adaptive large neighbourhood search metaheuristic.
          *Journal of Heuristics* (2018) 24 (5): 783–815.
        - Kirkpatrick, S., Gerlatt, C. D. Jr., and Vecchi, M. P. Optimization
          by Simulated Annealing. *IBM Research Report* RC 9355, 1982.
        """
        if start_temperature < 0 or end_temperature < 0 or step < 0:
            raise ValueError("Temperatures must be positive.")

        if start_temperature < end_temperature:
            raise ValueError("Start temperature must be bigger than end "
                             "temperature.")

        self._start_temperature = start_temperature
        self._end_temperature = end_temperature
        self._step = step

        self._temperature = start_temperature

    @property
    def start_temperature(self):
        return self._start_temperature

    @property
    def end_temperature(self):
        return self._end_temperature

    @property
    def step(self):
        return self._step

    def accept(self, rnd, best, current, candidate):
        probability = np.exp((current.objective() - candidate.objective())
                             / self._temperature)

        # We should not set a temperature that is lower than the end
        # temperature.
        self._temperature = max(self.end_temperature,
                                self._temperature - self.step)

        return probability >= rnd.random_sample()
