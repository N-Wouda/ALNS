import warnings
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import numpy.random as rnd

from .Result import Result
from .State import State
from .Statistics import Statistics
from .tools.warnings import OverwriteWarning
from .weight_schemes import WeightScheme

# Potential candidate solution consideration outcomes.
_BEST = 0
_BETTER = 1
_ACCEPTED = 2
_REJECTED = 3


class ALNS:

    def __init__(self, rnd_state: rnd.RandomState = rnd.RandomState()):
        """
        Implements the adaptive large neighbourhood search (ALNS) algorithm.
        The implementation optimises for a minimisation problem, as explained
        in the text by Pisinger and Røpke (2010).

        Parameters
        ----------
        rnd_state
            Optional random state to use for random number generation. When
            passed, this state is used for operator selection and general
            computations requiring random numbers. It is also passed to the
            destroy and repair operators, as a second argument.

        References
        ----------
        [1]: Pisinger, D., and Røpke, S. (2010). Large Neighborhood Search. In
             M. Gendreau (Ed.), *Handbook of Metaheuristics* (2 ed., pp. 399
             - 420). Springer.
        """
        self._destroy_operators = OrderedDict()
        self._repair_operators = OrderedDict()

        # Optional callback that may be used to improve a new best solution
        # further, via e.g. local search.
        self._on_best = None

        # Current and best solutions
        self._best = None
        self._curr = None

        self._rnd_state = rnd_state

    @property
    def destroy_operators(self) -> List[Tuple[str, Callable]]:
        """
        Returns the destroy operators set for the ALNS algorithm.

        Returns
        -------
        A list of (name, operator) tuples. Their order is the same as the one in
        which they were passed to the ALNS instance.
        """
        return list(self._destroy_operators.items())

    @property
    def repair_operators(self) -> List[Tuple[str, Callable]]:
        """
        Returns the repair operators set for the ALNS algorithm.

        Returns
        -------
        A list of (name, operator) tuples. Their order is the same as the one in
        which they were passed to the ALNS instance.
        """
        return list(self._repair_operators.items())

    def add_destroy_operator(self,
                             op: Callable[[State, rnd.RandomState], State],
                             name: Optional[str] = None):
        """
        Adds a destroy operator to the heuristic instance.

        Parameters
        ----------
        op
            An operator that, when applied to the current state, returns a new
            state reflecting its implemented destroy action. The second argument
            is the random state constructed from the passed-in seed.
        name
            Optional name argument, naming the operator. When not passed, the
            function name is used instead.
        """
        self._add_operator(self._destroy_operators, op, name)

    def add_repair_operator(self,
                            op: Callable[[State, rnd.RandomState], State],
                            name: Optional[str] = None):
        """
        Adds a repair operator to the heuristic instance.

        Parameters
        ----------
        op
            An operator that, when applied to the destroyed state, returns a
            new state reflecting its implemented repair action. The second
            argument is the random state constructed from the passed-in seed.
        name
            Optional name argument, naming the operator. When not passed, the
            function name is used instead.
        """
        self._add_operator(self._repair_operators, op, name)

    def iterate(self,
                init_sol: State,
                weight_scheme: WeightScheme,
                crit: Callable[[rnd.RandomState, State, State, State], bool],
                iters: int = 10_000,
                stats: Statistics = Statistics()) -> Result:
        """
        Runs the adaptive large neighbourhood search heuristic [1], using the
        previously set destroy and repair operators. The first solution is set
        to the passed-in initial solution, and then subsequent solutions are
        computed by iteratively applying the operators.

        Parameters
        ----------
        init_sol
            The initial solution, as a State object.
        weight_scheme
            The weight scheme to use for updating the (adaptive) weights. See
            also the ``alns.weight_schemes`` module for an overview.
        crit
            The acceptance criterion to use for candidate states. See also
            the ``alns.criteria`` module for an overview.
        iters
            The number of iterations. Default 10_000.
        stats
            Optional Statistics object.

        Raises
        ------
        ValueError
            When the parameters do not meet requirements.

        Returns
        -------
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
        if len(self.destroy_operators) == 0 or len(self.repair_operators) == 0:
            raise ValueError("Missing at least one destroy or repair operator.")

        if iters < 0:
            raise ValueError("Negative number of iterations.")

        self._curr = self._best = init_sol

        stats.collect_objective(init_sol.objective())

        for iteration in range(iters):
            weight_scheme.at_iteration_start(iteration, iters)

            d_idx, r_idx = weight_scheme.select_operators(self._rnd_state)

            d_name, d_operator = self.destroy_operators[d_idx]
            r_name, r_operator = self.repair_operators[r_idx]

            destroyed = d_operator(self._curr, self._rnd_state)
            cand = r_operator(destroyed, self._rnd_state)

            s_idx = self._consider_candidate(cand, crit)

            weight_scheme.update_weights(d_idx, r_idx, s_idx)

            stats.collect_objective(self._curr.objective())
            stats.collect_destroy_operator(d_name, s_idx)
            stats.collect_repair_operator(r_name, s_idx)

        return Result(self._best, stats)

    def on_best(self, func: Callable[[State, rnd.RandomState], State]):
        """
        Sets a callback function to be called when ALNS finds a new global best
        solution state.

        Parameters
        ----------
        func
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

    def _consider_candidate(self,
                            cand: State,
                            crit: Callable[..., bool]) -> int:
        """
        Considers the candidate solution by comparing it against the best and
        current solutions. Candidate solutions are accepted based on the
        passed-in acceptance criterion. The weight index (best, better,
        accepted, rejected) is returned.

        The best/current solutions are updated as a side-effect.

        Returns
        -------
        A weight index. This index indicates the consideration outcome.
        """
        w_idx = _REJECTED

        if crit(self._rnd_state, self._best, self._curr, cand):
            w_idx = (_BETTER
                     if cand.objective() < self._curr.objective()
                     else _ACCEPTED)

            self._curr = cand

        if cand.objective() < self._best.objective():  # is new best?
            if self._on_best:
                cand = self._on_best(cand, self._rnd_state)

            # New best solution becomes starting point in next iteration.
            self._best = cand
            self._current = cand

            return _BEST

        # Best has not been updated if we get here, but the current state might
        # have (if the candidate was accepted).
        return w_idx
