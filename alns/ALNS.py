import numpy as np
import numpy.random as rnd

from .Result import Result
from .State import State    # pylint: disable=unused-import
from .accept import accept
from .enums import WeightIndex


class ALNS:

    def __init__(self, rnd_state=rnd.RandomState()):
        """
        TODO

        Parameters
        ----------
        rnd_state : rnd.RandomState
            Optional random state to use for random number generation. When
            passed, this state is used for operator selection and general
            computations requiring random numbers. It is also passed to the
            destroy and repair operators, as a second argument.
        """
        self._destroy_operators = []
        self._repair_operators = []

        self._rnd_state = rnd_state
        self._iteration = 0

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

    def iterate(self, initial_solution, weights, operator_decay,
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
        **kwargs
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

        for iteration in range(iterations):
            self._iteration = iteration

            d_idx = self._rnd_state.choice(
                np.arange(0, len(self.destroy_operators)),
                p=d_weights / np.sum(d_weights))

            r_idx = self._rnd_state.choice(
                np.arange(0, len(self.repair_operators)),
                p=r_weights / np.sum(r_weights))

            destroyed = self.destroy_operators[d_idx](current, self._rnd_state)
            candidate = self.repair_operators[r_idx](destroyed, self._rnd_state)

            current, weight = self._consider_candidate(best, current, candidate,
                                                       weights, **kwargs)

            if current.objective() > best.objective():
                best = current

            # The weights are updated as convex combinations of the current
            # weight and the update parameter. See eq. (2), p. 12.
            d_weights[d_idx] *= operator_decay
            d_weights[d_idx] += (1 - operator_decay) * weight

            r_weights[r_idx] *= operator_decay
            r_weights[r_idx] += (1 - operator_decay) * weight

        return Result(best, current)

    def _consider_candidate(self, best, current, candidate, weights,
                            anneal=True, initial_temperature=10000,
                            temperature_decay=0.95):
        """
        Considers the candidate solution by comparing it against the best and
        current solutions. Returns the new solution when it is better or
        accepted, or the current in case it is rejected. An annealing-based
        acceptance criterion might be used, in which case worse solutions are
        accepted from time to time.

        Parameters
        ----------
        best : State
            Best solution encountered so far.
        current : State
            Current solution.
        candidate : State
            Candidate solution.
        weights : array_like
            Updating weights for the operators.
        anneal : bool
            Should an annealing approach be used when considering inferior
            candidate solutions? Defaults to True.
        initial_temperature : float
            The initial temperature. Defaults to 1000.
        temperature_decay : float
            Temperature decay parameter. Defaults to 0.95, in line with
            Kirkpatrick et al. (1982).

        Returns
        -------
        State
            The new state.
        float
            The weight value to use when updating the operator weights.

        References
        ----------
        Kirkpatrick, S., Gerlatt, C. D. Jr., and Vecchi, M. P., Optimization by
        Simulated Annealing, *IBM Research Report* RC 9355, 1982.
        """
        if not (0 < temperature_decay < 1):
            raise ValueError("Temperature decay parameter outside unit"
                             " interval is not understood.")

        if candidate.objective() > best.objective():
            return candidate, weights[WeightIndex.IS_BEST]

        if candidate.objective() > current.objective():
            return candidate, weights[WeightIndex.IS_BETTER]

        temperature = self._compute_temperature(initial_temperature,
                                                temperature_decay)

        # The temperature-based acceptance criterion allows accepting worse
        # solutions, especially in early iterations.
        if anneal and accept(current, candidate, temperature, self._rnd_state):
            return candidate, weights[WeightIndex.IS_ACCEPTED]

        return current, weights[WeightIndex.IS_REJECTED]

    def _compute_temperature(self, initial_temperature, temperature_decay):
        """
        For this particular updating scheme, see Kirkpatrick et al. (1982).
        """
        return initial_temperature * np.power(temperature_decay,
                                              self._iteration)

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
