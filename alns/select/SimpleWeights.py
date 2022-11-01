import logging
from typing import List, Optional

import numpy as np

from alns.select.WeightScheme import WeightScheme

logger = logging.getLogger(__name__)


class SimpleWeights(WeightScheme):
    """
    A simple weighting scheme, where the operator weights are adjusted
    continuously throughout the algorithm runs. This works as follows.
    In each iteration, the old weight is updated with a score based on a
    convex combination of the existing weight and the new score, as:

    ``new_weight = op_decay * old_weight + (1 - op_decay) * score``

    Parameters
    ----------
    (other arguments are explained in ``WeightScheme``)

    op_decay
        Decay parameter in [0, 1]. This parameter is used to weigh the
        running performance of each operator.
    """

    def __init__(
        self,
        scores: List[float],
        num_destroy: int,
        num_repair: int,
        op_decay: float,
        *,
        op_coupling: Optional[np.ndarray] = None
    ):
        super().__init__(
            scores, num_destroy, num_repair, op_coupling=op_coupling
        )

        if not (0 <= op_decay <= 1):
            raise ValueError("op_decay outside [0, 1] not understood.")

        self._op_decay = op_decay

    def update(self, d_idx, r_idx, s_idx):
        self._d_weights[d_idx] *= self._op_decay
        self._d_weights[d_idx] += (1 - self._op_decay) * self._scores[s_idx]

        self._r_weights[r_idx] *= self._op_decay
        self._r_weights[r_idx] += (1 - self._op_decay) * self._scores[s_idx]
