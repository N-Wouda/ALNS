from typing import Callable, List, Tuple

import numpy.random as rnd

from alns.Result import Result
from alns.State import State
from alns.Statistics import Statistics
from alns.criteria import AcceptanceCriterion
from alns.weight_schemes import WeightScheme

# Potential candidate solution consideration outcomes.
_BEST = 0
_BETTER = 1
_ACCEPTED = 2
_REJECTED = 3

_OperatorType = Callable[[State, rnd.RandomState], State]


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
        self._destroy_operators = {}
        self._repair_operators = {}

        # Optional callback that may be used to improve a new best solution
        # further, via e.g. local search.
        self._on_best = None

        # Current and best solutions
        self._best = None
        self._curr = None

        self._rnd_state = rnd_state

    @property
    def destroy_operators(self) -> List[Tuple[str, _OperatorType]]:
        """
        Returns the destroy operators set for the ALNS algorithm.

        Returns
        -------
        A list of (name, operator) tuples. Their order is the same as the one in
        which they were passed to the ALNS instance.
        """
        return list(self._destroy_operators.items())

    @property
    def repair_operators(self) -> List[Tuple[str, _OperatorType]]:
        """
        Returns the repair operators set for the ALNS algorithm.

        Returns
        -------
        A list of (name, operator) tuples. Their order is the same as the one in
        which they were passed to the ALNS instance.
        """
        return list(self._repair_operators.items())

    def add_destroy_operator(self, op: _OperatorType, name: str = None):
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
        self._destroy_operators[op.__name__ if name is None else name] = op

    def add_repair_operator(self, op: _OperatorType, name: str = None):
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
        self._repair_operators[name if name else op.__name__] = op

    def iterate(self,
                init_sol: State,
                weight_scheme: WeightScheme,
                crit: AcceptanceCriterion,
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

        stats.collect_destroy_weights(weight_scheme.destroy_weights)
        stats.collect_repair_weights(weight_scheme.repair_weights)

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

            stats.collect_destroy_weights(weight_scheme.destroy_weights)
            stats.collect_repair_weights(weight_scheme.repair_weights)

        return Result(self._best, stats)

    def on_best(self, func: _OperatorType):
        """
        Sets a callback function to be called when ALNS finds a new global best
        solution state.

        Parameters
        ----------
        func
            A function that should take a solution State as its first parameter,
            and a numpy RandomState as its second (cf. the operator signature).
            It should return a (new) solution State.
        """
        self._on_best = func

    def _consider_candidate(self,
                            cand: State,
                            crit: AcceptanceCriterion) -> int:
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

        if cand.objective() < self._best.objective():  # is new best
            if self._on_best:
                cand = self._on_best(cand, self._rnd_state)

            # New best solution becomes starting point in next iteration.
            self._best = cand
            self._curr = cand

            return _BEST

        return w_idx
