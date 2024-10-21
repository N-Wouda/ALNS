import logging
import time
from typing import Dict, List, Optional, Protocol, Tuple

import numpy.random as rnd

from alns.Outcome import Outcome
from alns.Result import Result
from alns.State import State
from alns.Statistics import Statistics
from alns.accept import AcceptanceCriterion
from alns.select import OperatorSelectionScheme
from alns.stop import StoppingCriterion


class _OperatorType(Protocol):
    __name__: str

    def __call__(
        self,
        state: State,
        rng: rnd.Generator,
        **kwargs,
    ) -> State:
        ...  # pragma: no cover


class _CallbackType(Protocol):
    __name__: str

    def __call__(self, state: State, rng: rnd.Generator, **kwargs):
        ...  # pragma: no cover


logger = logging.getLogger(__name__)


class ALNS:
    """
    Implements the adaptive large neighbourhood search (ALNS) algorithm.
    The implementation optimises for a minimisation problem, as explained
    in the text by Pisinger and Røpke (2010).

    .. note::

        Like the operators passed into the ALNS instance, any registered
        callback functions (registered via :meth:`~alns.ALNS.ALNS.on_best`,
        :meth:`~alns.ALNS.ALNS.on_better`, :meth:`~alns.ALNS.ALNS.on_accept`,
        or :meth:`~alns.ALNS.ALNS.on_reject`) should take a candidate
        :class:`~alns.State.State` and :class:`~numpy.random.Generator` as
        arguments. Unlike the operators, no solution should be returned: if
        desired, the given candidate solution should be modified in-place
        instead. Note that this solution is **not** evaluated again (so a
        rejected candidate solution will stay rejected!).

    Parameters
    ----------
    rng
        Optional random number generator (RNG). When passed, this generator
        is used for operator selection and general computations requiring
        random numbers. It is also passed to the destroy and repair operators,
        as a second argument.

    References
    ----------
    .. [1] Pisinger, D., and Røpke, S. (2010). Large Neighborhood Search. In
           M. Gendreau (Ed.), *Handbook of Metaheuristics* (2 ed., pp. 399
           - 420). Springer.
    """

    def __init__(self, rng: rnd.Generator = rnd.default_rng()):
        self._rng = rng

        self._d_ops: Dict[str, _OperatorType] = {}
        self._r_ops: Dict[str, _OperatorType] = {}

        # Registers callback for each possible evaluation outcome.
        self._on_outcome: Dict[Outcome, _CallbackType] = {}

    @property
    def destroy_operators(self) -> List[Tuple[str, _OperatorType]]:
        """
        Returns the destroy operators set for the ALNS algorithm.

        Returns
        -------
        list
            A list of (name, operator) tuples. Their order is the same as the
            one in which they were passed to the ALNS instance.
        """
        return list(self._d_ops.items())

    @property
    def repair_operators(self) -> List[Tuple[str, _OperatorType]]:
        """
        Returns the repair operators set for the ALNS algorithm.

        Returns
        -------
        list
            A list of (name, operator) tuples. Their order is the same as the
            one in which they were passed to the ALNS instance.
        """
        return list(self._r_ops.items())

    def add_destroy_operator(
        self, op: _OperatorType, name: Optional[str] = None
    ):
        """
        Adds a destroy operator to the heuristic instance.

        .. warning::

            A destroy operator will receive the current solution state
            maintained by the ALNS instance, not a copy. Make sure to modify
            a **copy** of this state in the destroy operator, created using,
            for example, :func:`copy.copy` or :func:`copy.deepcopy`.

        Parameters
        ----------
        op
            An operator that, when applied to the current state, returns a new
            state reflecting its implemented destroy action. Its second
            argument is the RNG passed to the ALNS instance.
        name
            Optional name argument, naming the operator. When not passed, the
            function name is used instead.
        """
        logger.debug(f"Adding destroy operator {op.__name__}.")
        self._d_ops[op.__name__ if name is None else name] = op

    def add_repair_operator(
        self, op: _OperatorType, name: Optional[str] = None
    ):
        """
        Adds a repair operator to the heuristic instance.

        Parameters
        ----------
        op
            An operator that, when applied to the destroyed state, returns a
            new state reflecting its implemented repair action. Its second
            argument is the RNG passed to the ALNS instance.
        name
            Optional name argument, naming the operator. When not passed, the
            function name is used instead.
        """
        logger.debug(f"Adding repair operator {op.__name__}.")
        self._r_ops[name if name else op.__name__] = op

    def iterate(
        self,
        initial_solution: State,
        op_select: OperatorSelectionScheme,
        accept: AcceptanceCriterion,
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
        op_select
            The operator selection scheme to use for selecting operators.
            See also the ``alns.select`` module for an overview.
        accept
            The acceptance criterion to use for candidate states.
            See also the ``alns.accept`` module for an overview.
        stop
            The stopping criterion to use for stopping the iterations.
            See also the ``alns.stop`` module for an overview.
        **kwargs
            Optional keyword arguments. These are passed to the operators and
            any registered callbacks.

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
        .. [1] Pisinger, D., & Røpke, S. (2010). Large Neighborhood Search. In
               M. Gendreau (Ed.), *Handbook of Metaheuristics* (2 ed., pp. 399
               - 420). Springer.

        .. [2] S. Røpke and D. Pisinger (2006). A unified heuristic for a large
               class of vehicle routing problems with backhauls. *European
               Journal of Operational Research*, 171: 750-775.
        """
        if len(self.destroy_operators) == 0 or len(self.repair_operators) == 0:
            raise ValueError("Missing destroy or repair operators.")

        curr = best = initial_solution
        init_obj = initial_solution.objective()

        logger.debug(f"Initial solution has objective {init_obj:.2f}.")

        stats = Statistics()
        stats.collect_objective(init_obj)
        stats.collect_runtime(time.perf_counter())

        while not stop(self._rng, best, curr):
            d_idx, r_idx = op_select(self._rng, best, curr)

            d_name, d_operator = self.destroy_operators[d_idx]
            r_name, r_operator = self.repair_operators[r_idx]

            logger.debug(f"Selected operators {d_name} and {r_name}.")

            destroyed = d_operator(curr, self._rng, **kwargs)
            cand = r_operator(destroyed, self._rng, **kwargs)

            best, curr, outcome = self._eval_cand(
                accept, best, curr, cand, **kwargs
            )

            op_select.update(cand, d_idx, r_idx, outcome)

            stats.collect_objective(curr.objective())
            stats.collect_destroy_operator(d_name, outcome)
            stats.collect_repair_operator(r_name, outcome)
            stats.collect_runtime(time.perf_counter())

        logger.info(f"Finished iterating in {stats.total_runtime:.2f}s.")

        return Result(best, stats)

    def on_best(self, func: _CallbackType):
        """
        Sets a callback function to be called when ALNS finds a new global best
        solution state.
        """
        logger.debug(f"Adding on_best callback {func.__name__}.")
        self._on_outcome[Outcome.BEST] = func

    def on_better(self, func: _CallbackType):
        """
        Sets a callback function to be called when ALNS finds a better solution
        than the current incumbent.
        """
        logger.debug(f"Adding on_better callback {func.__name__}.")
        self._on_outcome[Outcome.BETTER] = func

    def on_accept(self, func: _CallbackType):
        """
        Sets a callback function to be called when ALNS accepts a new solution
        as the current incumbent (that is not a new global best, or otherwise
        improving).
        """
        logger.debug(f"Adding on_accept callback {func.__name__}.")
        self._on_outcome[Outcome.ACCEPT] = func

    def on_reject(self, func: _CallbackType):
        """
        Sets a callback function to be called when ALNS rejects a new solution.
        """
        logger.debug(f"Adding on_reject callback {func.__name__}.")
        self._on_outcome[Outcome.REJECT] = func

    def _eval_cand(
        self,
        accept: AcceptanceCriterion,
        best: State,
        curr: State,
        cand: State,
        **kwargs,
    ) -> Tuple[State, State, Outcome]:
        """
        Considers the candidate solution by comparing it against the best and
        current solutions. Candidate solutions are accepted based on the
        passed-in acceptance criterion. The (possibly new) best and current
        solutions are returned, along with a weight index (best, better,
        accepted, rejected).

        Returns
        -------
        tuple
            A tuple of the best and current solution, along with the weight
            index.
        """
        outcome = self._determine_outcome(accept, best, curr, cand)
        func = self._on_outcome.get(outcome)

        if callable(func):
            func(cand, self._rng, **kwargs)

        if outcome == Outcome.BEST:
            return cand, cand, outcome

        if outcome == Outcome.REJECT:
            return best, curr, outcome

        return best, cand, outcome

    def _determine_outcome(
        self,
        accept: AcceptanceCriterion,
        best: State,
        curr: State,
        cand: State,
    ) -> Outcome:
        """
        Determines the candidate solution's evaluation outcome.
        """
        outcome = Outcome.REJECT

        if accept(self._rng, best, curr, cand):  # accept candidate
            outcome = Outcome.ACCEPT

            if cand.objective() < curr.objective():
                outcome = Outcome.BETTER

        if cand.objective() < best.objective():  # candidate is new best
            logger.info(f"New best with objective {cand.objective():.2f}.")
            outcome = Outcome.BEST

        return outcome
