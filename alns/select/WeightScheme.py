from abc import abstractmethod
from typing import List, Optional

import numpy as np

from alns.select.SelectionScheme import SelectionScheme


class WeightScheme(SelectionScheme):
    """
    Base class from which to implement an operator selection scheme based
    on weights.

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
    op_coupling
        Optional keyword argument. Matrix that indicates coupling between
        destroy and repair operators. Entry (i, j) is 1 if destroy operator i
        can be used in conjunction with repair operator j and 0 otherwise.
    """

    def __init__(
        self,
        scores: List[float],
        num_destroy: int,
        num_repair: int,
        *,
        op_coupling: Optional[np.ndarray] = None,
    ):
        super().__init__(num_destroy, num_repair, op_coupling=op_coupling)

        if any(score < 0 for score in scores):
            raise ValueError("Negative scores are not understood.")

        if len(scores) < 4:
            # More than four is OK because we only use the first four.
            raise ValueError(f"Expected four scores, found {len(scores)}")

        self._scores = scores
        self._d_weights = np.ones(num_destroy, dtype=float)
        self._r_weights = np.ones(num_repair, dtype=float)

    @property
    def destroy_weights(self) -> np.ndarray:
        return self._d_weights

    @property
    def repair_weights(self) -> np.ndarray:
        return self._r_weights

    @abstractmethod
    def update(self, d_idx: int, r_idx: int, s_idx: int):
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
