from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from numpy.random import RandomState


class WeightScheme(ABC):
    """
    Base class from which to implement a weight scheme.
    """

    def __init__(self, weights: List[float], num_destroy: int, num_repair: int):
        self._validate_arguments(weights, num_destroy, num_repair)

        self._weights = weights

        self._d_weights = np.ones(num_destroy, dtype=float)
        self._r_weights = np.ones(num_repair, dtype=float)

    def at_iteration_start(self, iteration: int, max_iterations: int):
        """
        A simple observer hook that is called at the start of each iteration.
        This may be used to update the weights/operator selection mechanism
        from time to time.

        Parameters
        ----------
        iteration
            The current iteration number.
        max_iterations
            Maximum number of iterations.
        """
        pass

    @abstractmethod
    def select_operators(self, rnd_state: RandomState) -> Tuple[int, int]:
        """
        Selects a destroy and repair operator pair to apply in this iteration.

        Parameters
        ----------
        rnd_state
            Random state object, to be used for number generation.

        Returns
        -------
        A tuple of (d_idx, r_idx), which are indices into the destroy and repair
        operator lists, respectively.
        """
        return NotImplemented

    @abstractmethod
    def update_weights(self, d_idx: int, r_idx: int, w_idx: int):
        """
        Updates the weights associated with the applied destroy (d_idx) and
        repair (r_idx) operators. The final weight index (w_idx) indicates the
        outcome.

        Parameters
        ----------
        d_idx
            Destroy operator index.
        r_idx
            Repair operator index
        w_idx
            Weight index.
        """
        return NotImplemented

    @staticmethod
    def _validate_arguments(weights, num_destroy, num_repair):
        if any(weight < 0 for weight in weights):
            raise ValueError("Negative weights are not understood")

        if len(weights) < 4:
            # More than four is not problematic, but we only use the first four.
            raise ValueError("Unsupported number of weights: expected 4, "
                             "found {0}".format(len(weights)))

        if num_destroy <= 0 or num_repair <= 0:
            raise ValueError("Missing at least one destroy or repair operator.")
