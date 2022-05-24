from collections import defaultdict
import logging
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import numpy.random as rnd

from alns.Result import Result
from alns.State import State
from alns.Statistics import Statistics
from alns.accept import AcceptanceCriterion
from alns.stop import StoppingCriterion
from alns.weights import WeightScheme

# Potential candidate solution consideration outcomes.
_BEST = 0
_BETTER = 1
_ACCEPT = 2
_REJECT = 3

# TODO this should become a Protocol to allow for kwargs. See also this issue:
#  https://stackoverflow.com/q/61569324/4316405.
_OperatorType = Callable[[State, rnd.RandomState], State]

logger = logging.getLogger(__name__)


class ALNS:
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

    def __init__(self, rnd_state: rnd.RandomState = rnd.RandomState()):
        self._d_ops: Dict[str, _OperatorType] = {}
        self._r_ops: Dict[str, _OperatorType] = {}
        self._only_after: Dict[_OperatorType, set] = defaultdict(set)

        # Optional callback that may be used to improve a new best solution
        # further, via e.g. local search.
        self._on_best: Optional[_OperatorType] = None

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
        return list(self._d_ops.items())

    @property
    def repair_operators(self) -> List[Tuple[str, _OperatorType]]:
        """
        Returns the repair operators set for the ALNS algorithm.

        Returns
        -------
        A list of (name, operator) tuples. Their order is the same as the one in
        which they were passed to the ALNS instance.
        """
        return list(self._r_ops.items())

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
        logger.debug(f"Adding destroy operator {op.__name__}.")
        self._d_ops[op.__name__ if name is None else name] = op

    def add_repair_operator(
        self,
        op: _OperatorType,
        name: str = None,
        *,
        only_after: Optional[Iterable[_OperatorType]] = None,
    ):
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
        only_after
            Optional keyword-only argument indicating which destroy operators
            work with the passed-in repair operator. If passed, this argument
            should be an iterable (e.g. a list) of destroy operators. If not
            passed, the default is to assume that all destroy operators work
            with the new repair operator.
        """
        logger.debug(f"Adding repair operator {op.__name__}.")
        self._r_ops[name if name else op.__name__] = op

        if only_after:
            self._only_after[op].update(only_after)

    def _compute_op_coupling(self) -> np.ndarray:
        """
        Internal helper to compute a matrix that describes the
        coupling between destroy and repair operators. The matrix has size
        |d_ops|-by-|r_ops| and entry (i, j) is 1 if destroy operator i can
        be used in conjunction with repair operator j and 0 otherwise.

        If the only_after keyword-only argument was not used when adding
        the repair operators, then all entries of the matrix are 1.
        """
        op_coupling = np.ones((len(self._d_ops), len(self._r_ops)))

        for r_idx, (_, r_op) in enumerate(self.repair_operators):
            coupled_d_ops = self._only_after[r_op]

            for d_idx, (_, d_op) in enumerate(self.destroy_operators):
                if coupled_d_ops and d_op not in coupled_d_ops:
                    op_coupling[d_idx, r_idx] = 0

        # Destroy operators must be coupled with at least one repair operator
        d_idcs = np.flatnonzero(np.count_nonzero(op_coupling, axis=1) == 0)

        if d_idcs.size != 0:
            d_name, _ = self.destroy_operators[d_idcs[0]]
            raise ValueError(f"{d_name} has no coupled repair operators.")

        return op_coupling

    def iterate(
        self,
        initial_solution: State,
        weight_scheme: WeightScheme,
        crit: AcceptanceCriterion,
        stop: StoppingCriterion,
        **kwargs,
    ) -> Result:
        """
        Runs the adaptive large neighbourhood search heuristic [1], using the
        previously set destroy and repair operators. The first solution is set
        to the passed-in initial solution, and then subsequent solutions are
        computed by iteratively applying the operators.

        Parameters
        ----------
        initial_solution
            The initial solution, as a State object.
        weight_scheme
            The weight scheme to use for updating the (adaptive) weights. See
            also the ``alns.weight_schemes`` module for an overview.
        crit
            The acceptance criterion to use for candidate states. See also
            the ``alns.criteria`` module for an overview.
        stop
            The stopping criterion to use for stopping the iterations.
            See also the ``alns.stopping_criteria`` module for an overview.

        **kwargs
            Optional keyword arguments. These are passed to the operators,
            including callbacks.

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
            raise ValueError("Missing destroy or repair operators.")

        curr = best = initial_solution
        init_obj = initial_solution.objective()
        op_coupling = self._compute_op_coupling()

        logger.debug(f"Initial solution has objective {init_obj:.2f}.")

        stats = Statistics()
        stats.collect_objective(init_obj)
        stats.collect_runtime(time.perf_counter())

        while not stop(self._rnd_state, best, curr):
            d_idx, r_idx = weight_scheme.select_operators(
                self._rnd_state, op_coupling
            )

            d_name, d_operator = self.destroy_operators[d_idx]
            r_name, r_operator = self.repair_operators[r_idx]

            logger.debug(f"Selected operators {d_name} and {r_name}.")

            destroyed = d_operator(curr, self._rnd_state, **kwargs)
            cand = r_operator(destroyed, self._rnd_state, **kwargs)

            best, curr, s_idx = self._eval_cand(
                crit, best, curr, cand, **kwargs
            )

            weight_scheme.update_weights(d_idx, r_idx, s_idx)

            stats.collect_objective(curr.objective())
            stats.collect_destroy_operator(d_name, s_idx)
            stats.collect_repair_operator(r_name, s_idx)
            stats.collect_runtime(time.perf_counter())

        logger.info(f"Finished iterating in {stats.total_runtime:.2f}s.")

        return Result(best, stats)

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
        logger.debug(f"Adding on_best callback {func.__name__}.")
        self._on_best = func

    def _eval_cand(
        self,
        crit: AcceptanceCriterion,
        best: State,
        curr: State,
        cand: State,
        **kwargs,
    ) -> Tuple[State, State, int]:
        """
        Considers the candidate solution by comparing it against the best and
        current solutions. Candidate solutions are accepted based on the
        passed-in acceptance criterion. The (possibly new) best and current
        solutions are returned, along with a weight index (best, better,
        accepted, rejected).

        Returns
        -------
        A tuple of the best and current solution, along with the weight index.
        """
        w_idx = _REJECT

        if crit(self._rnd_state, best, curr, cand):  # accept candidate
            w_idx = _BETTER if cand.objective() < curr.objective() else _ACCEPT
            curr = cand

        if cand.objective() < best.objective():  # candidate is new best
            logger.info(f"New best with objective {cand.objective():.2f}.")

            if self._on_best:
                cand = self._on_best(cand, self._rnd_state, **kwargs)

            return cand, cand, _BEST

        return best, curr, w_idx
