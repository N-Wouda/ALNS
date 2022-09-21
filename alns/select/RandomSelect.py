from typing import Tuple

from numpy.random import RandomState
from alns.select.SelectionScheme import SelectionScheme


class RandomSelect(SelectionScheme):
    """
    A random selection scheme. At each iteration, a random destroy and repair
    operator is selected.
    """

    def __init__(self, num_destroy: int, num_repair: int):
        if num_destroy <= 0 or num_repair <= 0:
            raise ValueError(
                "Number of destroy or repair operators is non-positive."
            )

        self._num_destroy = num_destroy
        self._num_repair = num_repair

    @property
    def num_destroy(self) -> int:
        return self._num_destroy

    @property
    def num_repair(self) -> int:
        return self._num_repair

    def select_operators(self, rnd_state: RandomState) -> Tuple[int, int]:
        """
        Select a random destroy and repair operator.
        """
        d_idx = rnd_state.randint(self.num_destroy)
        r_idx = rnd_state.randint(self.num_repair)

        return d_idx, r_idx

    def update(self, d_idx: int, r_idx: int, s_idx: int):
        pass
