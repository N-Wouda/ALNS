from abc import ABC, abstractmethod
from typing import List, Tuple

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
    """

    def __init__(self, num_destroy: int, num_repair: int, *args):
        self._d_weights = np.ones(num_destroy, dtype=float)
        self._r_weights = np.ones(num_repair, dtype=float)

    @abstractmethod
    def select_operators(self, rnd_state: RandomState) -> Tuple[int, int]:
        """
        Selects a destroy and repair operator pair to apply in this iteration.

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
