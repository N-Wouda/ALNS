import numpy as np
import numpy.random as rnd

from .Result import Result
from .State import State    # pylint: disable=unused-import
from .accept import accept
from .enums import WeightIndex


class ALNS:

    def __init__(self, seed=None):
        """

        Parameters
        ----------
        seed : int
            The random seed to use for method and acceptance selection. This
            seed is also passed to the destroy and repair operators, as a
            second argument.
        """
        self._destroy_operators = []
        self._repair_operators = []

        self._random = rnd.RandomState(seed)

    @property
    def destroy_operators(self):
        return self._destroy_operators

    @property
    def repair_operators(self):
        return self._repair_operators

    def add_destroy_operator(self, operator):
        """
        Adds a destroy operator to the heuristic instance.

        Parameters
        ----------
        operator : Callable[[State, RandomState], State]
            An operator that, when applied to the current state, returns a new
            state reflecting its implemented destroy action. The second argument
            is the random state constructed from the passed-in seed.
        """
        self._destroy_operators.append(operator)

    def add_repair_operator(self, operator):
        """
        Adds a repair operator to the heuristic instance.

        Parameters
        ----------
        operator : Callable[[State, RandomState], State]
            An operator that, when applied to the destroyed state, returns a
            new state reflecting its implemented repair action. The second
            argument is the random state constructed from the passed-in seed.
        """
        self._repair_operators.append(operator)

    def __call__(self, initial_solution, weights, operator_decay,
                 iterations=10000, **kwargs):
        """
        Runs the adaptive large neighbourhood search heuristic [1], using the
        previously set destroy and repair operators. The first solution is set
        to the passed-in initial solution, and then subsequent solutions are
        computed by iteratively applying the operators.

        Parameters
        ----------
        initial_solution : State
            The initial solution, as a State object.
        weights: array_like
            A list of four positive elements, representing the weight updates
            when the candidate solution results in a new global best (idx 0),
            is better than the current solution (idx 1), the solution is
            accepted (idx 2), or rejected (idx 3).
        operator_decay : float
            The operator decay parameter, as a float in the unit interval.
        iterations : int
            The number of iterations. Default 10000.
        **kwargs: dict
            Arguments passed to determine the correct updating strategy. See
            code for details.

        Raises
        ------
        ValueError
            When the parameters do not meet requirements.

        Returns
        -------
        Result
            A result object, containing the best and last solutions, and some
            additional results.

        References
        ----------
        [1]: Pisinger, D., & Røpke, S. (2010). Large Neighborhood Search. In M.
        Gendreau (Ed.), *Handbook of Metaheuristics* (2 ed., pp. 399-420).
        Springer.

        [2]: S. Røpke and D. Pisinger (2006). A unified heuristic for a large
        class of vehicle routing problems with backhauls. *European Journal of
        Operational Research*, 171: 750–775, 2006.
        """
        weights = np.asarray(weights, dtype=np.float16)
        self._validate_parameters(weights, operator_decay, iterations)

        current = best = initial_solution
        d_weights = np.ones_like(self.destroy_operators, dtype=np.float16)
        r_weights = np.ones_like(self.repair_operators, dtype=np.float16)

        for _ in range(iterations):
            d_idx = self._random.choice(
                np.arange(0, len(self.destroy_operators)),
                p=d_weights / np.sum(d_weights))

            r_idx = self._random.choice(
                np.arange(0, len(self.repair_operators)),
                p=r_weights / np.sum(r_weights))

            destroyed = self.destroy_operators[d_idx](current, self._random)
            candidate = self.repair_operators[r_idx](destroyed, self._random)

            current, weight = self._update(best, current, candidate, weights,
                                           **kwargs)

            # The weights are updated as convex combinations of the current
            # weight and the update parameter. See eq. (2), p. 12.
            d_weights[d_idx] *= operator_decay
            d_weights[d_idx] += (1 - operator_decay) * weight

            r_weights[r_idx] *= operator_decay
            r_weights[r_idx] += (1 - operator_decay) * weight

        return Result(best, current)

    def _update(self, best, current, candidate, weights, anneal=True,
                temperature=1000, temperature_decay=0.95):
        """
        TODO

        Returns
        -------
        State
            The new state.
        float
            The weight value to use when updating the operator weights.
        """
        if candidate.objective() > best.objective():
            return candidate, weights[WeightIndex.IS_BEST]

        if candidate.objective() > current.objective():
            return candidate, weights[WeightIndex.IS_BETTER]

        # The temperature-based acceptance criterion allows accepting worse
        # solutions, especially in early iterations.
        if anneal and accept(current, candidate, temperature,
                             temperature_decay, self._random):
            return candidate, weights[WeightIndex.IS_ACCEPTED]

        return current, weights[WeightIndex.IS_REJECTED]

    def _validate_parameters(self, weights, operator_decay, iterations):
        """
        Helper method to validate the passed-in ALNS parameters.
        """
        if not len(self.destroy_operators) or not len(self.repair_operators):
            raise ValueError("Missing at least one destroy or repair operator.")

        if not (0 < operator_decay < 1):
            raise ValueError("Operator decay parameter outside unit interval"
                             " is not understood.")

        if any(weight <= 0 for weight in weights):
            raise ValueError("Non-positive weights are not understood.")

        if len(weights) < 4:
            raise ValueError("Unsupported number of weights: expected 4,"
                             " found {0}.".format(len(weights)))

        if iterations <= 0:
            raise ValueError("Non-positive number of iterations.")
