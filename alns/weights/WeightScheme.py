from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from numpy.random import RandomState


class WeightScheme(ABC):
    """
    Base class from which to implement a weight scheme.

    Parameters
    ----------
    scores
        A list of four non-negative elements, representing the weight
        updates when the candidate solution results in a new global best
        (idx 0), is better than the current solution (idx 1), the solution
        is accepted (idx 2), or rejected (idx 3).
    num_destroy
        Number of destroy operators.
    num_repair
        Number of repair operators.
    """

    def __init__(self, scores: List[float], num_destroy: int, num_repair: int):
        self._validate_arguments(scores, num_destroy, num_repair)

        self._scores = scores
        self._d_weights = np.ones(num_destroy, dtype=float)
        self._r_weights = np.ones(num_repair, dtype=float)

    @property
    def destroy_weights(self) -> np.ndarray:
        return self._d_weights

    @property
    def repair_weights(self) -> np.ndarray:
        return self._r_weights

    def select_operators(
        self, rnd_state: RandomState, op_coupling: np.ndarray
    ) -> Tuple[int, int]:
        """
        Selects a destroy and repair operator pair to apply in this iteration.
        The default implementation uses a roulette wheel mechanism, where each
        operator is selected based on the normalised weights.

        Parameters
        ----------
        rnd_state
            Random state object, to be used for random number generation.
        op_coupling
            Matrix that indicates coupling between destroy and repair
            operators. Entry (i, j) is 1 if destroy operator i can be used in
            conjunction with repair operator j and 0 otherwise.

        Returns
        -------
        A tuple of (d_idx, r_idx), which are indices into the destroy and
        repair operator lists, respectively.
        """

        def select(op_weights):
            probs = op_weights / np.sum(op_weights)
            return rnd_state.choice(range(len(op_weights)), p=probs)

        d_idx = select(self._d_weights)
        coupled_r_idcs = np.flatnonzero(op_coupling[d_idx])
        r_idx = coupled_r_idcs[select(self._r_weights[coupled_r_idcs])]

        return d_idx, r_idx

    @abstractmethod
    def update_weights(self, d_idx: int, r_idx: int, s_idx: int):
        """
        Updates the weights associated with the applied destroy (d_idx) and
        repair (r_idx) operators. The score index (s_idx) indicates the
        outcome.

        Parameters
        ----------
        d_idx
            Destroy operator index.
        r_idx
            Repair operator index
        s_idx
            Score index.
        """
        return NotImplemented

    @staticmethod
    def _validate_arguments(scores, num_destroy, num_repair):
        if any(score < 0 for score in scores):
            raise ValueError("Negative scores are not understood.")

        if len(scores) < 4:
            # More than four is not problematic, but we only use the first four.
            raise ValueError(f"Expected four scores, found {len(scores)}")

        if num_destroy <= 0 or num_repair <= 0:
            raise ValueError("Missing destroy or repair operators.")
