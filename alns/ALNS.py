import warnings
from collections import OrderedDict

import numpy as np
import numpy.random as rnd

from .Result import Result
from .State import State  # pylint: disable=unused-import
from .Statistics import Statistics
from .Weights import Weights
from .criteria import AcceptanceCriterion  # pylint: disable=unused-import
from .select_operator import select_operator
from .tools.warnings import OverwriteWarning

# Weights
_IS_BEST = 0
_IS_BETTER = 1
_IS_ACCEPTED = 2
_IS_REJECTED = 3


class ALNS:

    def __init__(self, rnd_state=rnd.RandomState()):
        """
        Implements the adaptive large neighbourhood search (ALNS) algorithm.
        The implementation optimises for a minimisation problem, as explained
        in the text by Pisinger and Røpke (2010).

        Parameters
        ----------
        rnd_state : rnd.RandomState
            Optional random state to use for random number generation. When
            passed, this state is used for operator selection and general
            computations requiring random numbers. It is also passed to the
            destroy and repair operators, as a second argument.

        References
        ----------
        - Pisinger, D., and Røpke, S. (2010). Large Neighborhood Search. In M.
          Gendreau (Ed.), *Handbook of Metaheuristics* (2 ed., pp. 399-420).
          Springer.
        """
        super().__init__()

        self._destroy_operators = OrderedDict()
        self._repair_operators = OrderedDict()

        # These are (optional) callbacks
        self._on_best = None
        self._after_iteration = None

        # Current and best solutions
        self._best = None
        self._curr = None

        # Operator weights
        self._weights = None

        self._rnd_state = rnd_state

    @property
    def destroy_operators(self):
        """
        Returns the destroy operators set for the ALNS algorithm.

        Returns
        -------
        list
            A list of (name, operator) tuples. Their order is the same as the
            one in which they were passed to the ALNS instance.
        """
        return list(self._destroy_operators.items())

    @property
    def repair_operators(self):
        """
        Returns the repair operators set for the ALNS algorithm.

        Returns
        -------
        list
            A list of (name, operator) tuples. Their order is the same as the
            one in which they were passed to the ALNS instance.
        """
        return list(self._repair_operators.items())

    def add_destroy_operator(self, operator, name=None):
        """
        Adds a destroy operator to the heuristic instance.

        Parameters
        ----------
        operator : Callable[[State, RandomState], State]
            An operator that, when applied to the current state, returns a new
            state reflecting its implemented destroy action. The second argument
            is the random state constructed from the passed-in seed.
        name : str
            Optional name argument, naming the operator. When not passed, the
            function name is used instead.
        """
        self._add_operator(self._destroy_operators, operator, name)

    def add_repair_operator(self, operator, name=None):
        """
        Adds a repair operator to the heuristic instance.

        Parameters
        ----------
        operator : Callable[[State, RandomState], State]
            An operator that, when applied to the destroyed state, returns a
            new state reflecting its implemented repair action. The second
            argument is the random state constructed from the passed-in seed.
        name : str
            Optional name argument, naming the operator. When not passed, the
            function name is used instead.
        """
        self._add_operator(self._repair_operators, operator, name)

    def iterate(self, initial_solution, weights, op_decay, crit,
                iterations=10000, collect_stats=True):
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
            A list of four non-negative elements, representing the weight
            updates when the candidate solution results in a new global best
            (idx 0), is better than the current solution (idx 1), the solution
            is accepted (idx 2), or rejected (idx 3).
        op_decay : float
            The operator decay parameter, as a float in the unit interval,
            [0, 1] (inclusive).
        crit : AcceptanceCriterion
            The acceptance criterion to use for candidate states. See also
            the `alns.criteria` module for an overview.
        iterations : int
            The number of iterations. Default 10000.
        collect_stats : bool
            Should statistics be collected during iteration? Default True, but
            may be turned off for long runs to reduce memory consumption.

        Raises
        ------
        ValueError
            When the parameters do not meet requirements.

        Returns
        -------
        Result
            A result object, containing the best solution and some additional
            statistics.

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

        self._validate_parameters(weights, op_decay, iterations)

        self._curr = self._best = initial_solution
        self._weights = Weights(len(self.destroy_operators),
                                len(self.repair_operators))

        stats = Statistics()

        if collect_stats:
            stats.collect_objective(initial_solution.objective())

        for iteration in range(iterations):
            d_idx = select_operator(self.destroy_operators,
                                    self._weights.d_weights,
                                    self._rnd_state)

            r_idx = select_operator(self.repair_operators,
                                    self._weights.r_weights,
                                    self._rnd_state)

            d_name, d_operator = self.destroy_operators[d_idx]
            destroyed = d_operator(self._curr, self._rnd_state)

            r_name, r_operator = self.repair_operators[r_idx]
            cand = r_operator(destroyed, self._rnd_state)

            self._best, self._curr, w_idx = self._consider_candidate(cand, crit)

            # The weights are updated as convex combinations of the current
            # weight and the update parameter. See eq. (2), p. 12.
            self._weights.update_destroy(d_idx, op_decay, weights[w_idx])
            self._weights.update_repair(r_idx, op_decay, weights[w_idx])

            if collect_stats:
                stats.collect_objective(self._curr.objective())
                stats.collect_destroy_operator(d_name, w_idx)
                stats.collect_repair_operator(r_name, w_idx)

            if self._after_iteration:
                self._after_iteration()

        return Result(self._best, stats if collect_stats else None)

    def on_best(self, func):
        """
        Sets a callback function to be called when ALNS finds a new global best
        solution state.

        Parameters
        ----------
        func : callable
            A function that should take a solution State as its first parameter,
            and a numpy RandomState as its second (cf. the operator signature).
            It should return a (new) solution State.

        Warns
        -----
        OverwriteWarning
            When a callback has already been set.
        """
        if self._on_best:
            warnings.warn("A callback function has already been set to be "
                          "performed when a new best solution has been found."
                          " This callback will now be replaced by the newly"
                          " passed-in callback.",
                          OverwriteWarning)

        self._on_best = func

    def after_iterations(self, func):
        pass

    @staticmethod
    def _add_operator(operators, operator, name=None):
        """
        Internal helper that adds an operator to the passed-in operator
        dictionary. See `add_destroy_operator` and `add_repair_operator` for
        public methods that use this helper.

        Parameters
        ----------
        operators : dict
            Dictionary of (name, operator) key-value pairs.
        operator : Callable[[State, RandomState], State]
            Callable operator function.
        name : str
            Optional operator name.

        Warns
        -----
        OverwriteWarning
            When the operator name already maps to an operator on this ALNS
            instance.
        """
        if name is None:
            name = operator.__name__

        if name in operators:
            warnings.warn("The ALNS instance already knows an operator by the"
                          " name `{0}'. This operator will now be replaced with"
                          " the newly passed-in operator. If this is not what"
                          " you intended, consider explicitly naming your"
                          " operators via the `name' argument.".format(name),
                          OverwriteWarning)

        operators[name] = operator

    def _consider_candidate(self, candidate, criterion):
        """
        Considers the candidate solution by comparing it against the best and
        current solutions. Returns the new solution when it is better or
        accepted, or the current in case it is rejected. Candidate solutions
        are accepted based on the passed-in acceptance criterion.

        Parameters
        ----------
        candidate : State
            Candidate solution.
        criterion : AcceptanceCriterion
            The chosen acceptance criterion.

        Returns
        -------
        State
            The (possibly new) best state.
        State
            The (possibly new) current state.
        int
            The weight index to use when updating the operator weights.
        """
        weight = _IS_REJECTED

        if criterion.accept(self._rnd_state, self._best, self._curr, candidate):
            weight = (_IS_BETTER
                      if candidate.objective() < self._curr.objective()
                      else _IS_ACCEPTED)

            current = candidate

        if candidate.objective() < self._best.objective():  # is new best?
            if self._on_best:
                candidate = self._on_best(candidate, self._rnd_state)

            # New best solution becomes starting point in next iteration.
            return candidate, candidate, _IS_BEST

        # Best has not been updated if we get here, but the current state might
        # have (if the candidate was accepted).
        return self._best, self._curr, weight

    def _validate_parameters(self, weights, operator_decay, iterations):
        """
        Helper method to validate the passed-in ALNS parameters.
        """
        if len(self.destroy_operators) == 0 or len(self.repair_operators) == 0:
            raise ValueError("Missing at least one destroy or repair operator.")

        if not (0 <= operator_decay <= 1):
            raise ValueError("Operator decay parameter outside unit interval"
                             " is not understood.")

        if any(weight < 0 for weight in weights):
            raise ValueError("Negative weights are not understood.")

        if len(weights) < 4:
            # More than four is not explicitly problematic, as we only use the
            # first four anyways.
            raise ValueError("Unsupported number of weights: expected 4,"
                             " found {0}.".format(len(weights)))

        if iterations < 0:
            raise ValueError("Negative number of iterations.")
