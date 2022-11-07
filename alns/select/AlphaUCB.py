from typing import Optional

import numpy as np

from alns.select.OperatorSelectionScheme import OperatorSelectionScheme


class AlphaUCB(OperatorSelectionScheme):
    """
    :math:`\\alpha`-UCB (upper confidence bound) bandit scheme of Hendel
    (2022).

    TODO

    Parameters
    ----------
    alpha
        The :math:`\\alpha` parameter controls the width of the confidence
        interval. Larger values force the algorithm to select inferior
        operators more frequently, resulting in more exploration.
        :math:`\\alpha` must be in [0, 1].
    num_destroy
        Number of destroy operators.
    num_repair
        Number of repair operators.
    op_coupling
        Optional matrix that indicates coupling between destroy and repair
        operators. Entry (i, j) is 1 if destroy operator i can be used in
        conjunction with repair operator j and 0 otherwise.

    References
    ----------
    .. [1] Hendel, G. 2022. Adaptive large neighborhood search for mixed
           integer programming. *Mathematical Programming Computation* 14:
           185 – 221.
    """

    def __init__(
            self,
            alpha: float,
            num_destroy: int,
            num_repair: int,
            op_coupling: Optional[np.ndarray] = None,
    ):
        super().__init__(num_destroy, num_repair, op_coupling)

        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha outside [0, 1] not understood.")

        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    def __call__(self, rnd, best, curr):
        pass  # TODO

    def update(self, candidate, d_idx, r_idx, s_idx):
        pass  # TODO
