import numpy as np
import numpy.random as rnd

from .Result import Result
from .State import State


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

    def __call__(self, initial_solution, iterations=10000):
        """
        Runs the adaptive large neighbourhood search heuristic, using the
        previously set destroy and repair operators. The first solution is set
        to the passed-in initial solution, and then subsequent solutions are
        computed by iteratively applying the operators.

        Parameters
        ----------
        initial_solution : State
            The initial solution, as a State object
        iterations : int
            The number of iterations

        Returns
        -------
        Result
            A result object, containing the best and last solutions, and some
            additional results.
        """
        current_state = best_state = initial_solution
        d_weights = np.ones_like(self.destroy_operators)
        r_weights = np.ones_like(self.repair_operators)

        for iteration in range(iterations):
            d_idx = rnd.choice(np.arange(0, len(self.destroy_operators)),
                               p=d_weights / np.sum(d_weights))
            r_idx = rnd.choice(np.arange(0, len(self.repair_operators)),
                               p=r_weights / np.sum(r_weights))

            destroyed = self.destroy_operators[d_idx](current_state)
            next_state = self.repair_operators[r_idx](destroyed)

            if next_state.objective > best_state.objective:
                best_state = next_state
            elif next_state.objective > current_state.objective:
                current_state = next_state

            # TODO set weights

        return Result(best_state, current_state)
