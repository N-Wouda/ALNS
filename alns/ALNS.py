import numpy as np
import numpy.random as rnd

from .Result import Result
from .enums import WeightIndex


class ALNS:

    def __init__(self):
        self._destroy_operators = []
        self._repair_operators = []

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
        operator : Callable[[State], State]
            An operator that, when applied to the current state, returns a new
            state reflecting its implemented destroy action.
        """
        self._destroy_operators.append(operator)

    def add_repair_operator(self, operator):
        """
        Adds a repair operator to the heuristic instance.

        Parameters
        ----------
        operator : Callable[[State], State]
            An operator that, when applied to the current state, returns a new
            state reflecting its implemented repair action.
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
            The initial solution, as a State object
        weights: array_like
            A list of four elements, representing the weights attached to
            whether the candidate solution results in a new global best
            (idx 0), is better than the current solution (idx 1), the solution
            is accepted (idx 2), or rejected (idx 3).
        operator_decay : float
            TODO
        iterations : int
            The number of iterations. Default 10000.
        **kwargs: dict
            Arguments passed to determine the correct updating strategy. See
            code for details.

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
        current = best = initial_solution
        d_weights = np.ones_like(self.destroy_operators)
        r_weights = np.ones_like(self.repair_operators)

        for _ in range(iterations):
            d_idx = rnd.choice(np.arange(0, len(self.destroy_operators)),
                               p=d_weights / np.sum(d_weights))
            r_idx = rnd.choice(np.arange(0, len(self.repair_operators)),
                               p=r_weights / np.sum(r_weights))

            destroyed = self.destroy_operators[d_idx](current)
            candidate = self.repair_operators[r_idx](destroyed)

            current, weight = self._update(best, current, candidate, weights,
                                           **kwargs)

            # TODO set weights

        return Result(best, current)

    def _update(self, best, current, candidate, weights, anneal=True,
                temperature=1000, temperature_decay=0.9,):
        """
        TODO

        Returns
        -------
        State
            The new state
        float
            The weight value to use when updating the operator weights
        """
        if candidate.objective() > best.objective():
            return candidate, weights[WeightIndex.IS_BEST]

        if candidate.objective() > current.objective():
            return candidate, weights[WeightIndex.IS_BETTER]

        # The temperature-based acceptance criterion allows accepting worse
        # solutions, especially in early iterations.
        if anneal and self._accept(current, candidate, temperature,
                                   temperature_decay):
            return candidate, weights[WeightIndex.IS_ACCEPTED]

        return current, weights[WeightIndex.IS_REJECTED]

    def _accept(self, current, candidate, temperature, temperature_decay):
        """
        TODO

        Returns
        -------
        bool
            True is the candidate solution should be accepted, False if it
            should be rejected.
        """
        pass

