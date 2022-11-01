from abc import abstractmethod
from typing import List, Tuple, Optional

import numpy as np
from numpy.random import RandomState

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
        self._validate_arguments(scores, num_destroy, num_repair, op_coupling)

        self._scores = scores
        self._d_weights = np.ones(num_destroy, dtype=float)
        self._r_weights = np.ones(num_repair, dtype=float)

        self._op_coupling = (
            op_coupling
            if op_coupling is not None
            else np.ones((num_destroy, num_repair))
        )

    @property
    def destroy_weights(self) -> np.ndarray:
        return self._d_weights

    @property
    def repair_weights(self) -> np.ndarray:
        return self._r_weights

    @property
    def operator_coupling(self) -> np.ndarray:
        return self._op_coupling

    def select_operators(self, rnd_state: RandomState) -> Tuple[int, int]:
        """
        Selects a destroy and repair operator pair to apply in this iteration.
        The default implementation uses a roulette wheel mechanism, where each
        operator is selected based on the normalised weights.

        Parameters
        ----------
        rnd_state
            Random state object, to be used for random number generation.

        Returns
        -------
        A tuple of (d_idx, r_idx), which are indices into the destroy and
        repair operator lists, respectively.
        """

        def select(op_weights):
            probs = op_weights / np.sum(op_weights)
            return rnd_state.choice(range(len(op_weights)), p=probs)

        d_idx = select(self._d_weights)
        coupled_r_idcs = np.flatnonzero(self.operator_coupling[d_idx])
        r_idx = coupled_r_idcs[select(self._r_weights[coupled_r_idcs])]

        return d_idx, r_idx

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

    @staticmethod
    def _validate_arguments(scores, num_destroy, num_repair, op_coupling=None):
        if any(score < 0 for score in scores):
            raise ValueError("Negative scores are not understood.")

        if len(scores) < 4:
            # More than four is OK because we only use the first four.
            raise ValueError(f"Expected four scores, found {len(scores)}")

        if num_destroy <= 0 or num_repair <= 0:
            raise ValueError("Missing destroy or repair operators.")

        if op_coupling is None:
            return

        if op_coupling.shape != (num_destroy, num_repair):
            raise ValueError(
                "Op. coupling dimensions do not match num_destroy or num_repair."
            )

        d_idcs = np.flatnonzero(np.count_nonzero(op_coupling, axis=1) == 0)

        if d_idcs.size != 0:
            raise ValueError(
                f"Destroy ops. must be coupled with >= 1 repair operator."
            )
