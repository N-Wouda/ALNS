from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np
from numpy.random import RandomState


class SelectionScheme(ABC):
    """
    Base class from which to implement an operator selection scheme.

    Parameters
    ----------
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
        num_destroy: int,
        num_repair: int,
        *,
        op_coupling: Optional[np.ndarray] = None,
    ):
        self._validate_arguments(num_destroy, num_repair, op_coupling)

        self._num_destroy = num_destroy
        self._num_repair = num_repair
        self._op_coupling = (
            op_coupling
            if op_coupling is not None
            else np.ones((num_destroy, num_repair))
        )

    @property
    def num_destroy(self) -> int:
        return self._num_destroy

    @property
    def num_repair(self) -> int:
        return self._num_repair

    @property
    def op_coupling(self) -> np.ndarray:
        return self._op_coupling

    @abstractmethod
    def __call__(self, rnd: RandomState) -> Tuple[int, int]:
        """
        Determine which destroy and repair operator pair to apply in this
        iteration.

        Parameters
        ----------
        rnd_state
            Random state object, to be used for random number generation.

        Returns
        -------
        A tuple of (d_idx, r_idx), which are indices into the destroy and
        repair operator lists, respectively.
        """
        return NotImplemented

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
    def _validate_arguments(num_destroy, num_repair, op_coupling):
        if num_destroy <= 0 or num_repair <= 0:
            raise ValueError("Missing destroy or repair operators.")

        if op_coupling is None:
            return

        # Destroy ops. must be coupled with at least one repair operator
        d_idcs = np.flatnonzero(np.count_nonzero(op_coupling, axis=1) == 0)

        if d_idcs.size != 0:
            d_op = f"Destroy op. {d_idcs[0]}"
            raise ValueError(f"{d_op} has no coupled repair operators.")
