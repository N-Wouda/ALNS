import logging
from typing import List, Optional, Tuple

import numpy as np
from numpy.random import RandomState

from alns.select.WeightScheme import WeightScheme

logger = logging.getLogger(__name__)


class RouletteWheel(WeightScheme):
    """
    A selection scheme based on the roulette wheel mechanism, where each
    operator is selected based on normalised weights. The operator weights
    are adjusted continuously throughout the algorithm runs. This works as
    follows. In each iteration, the old weight is updated with a score based
    on a convex combination of the existing weight and the new score, as:

    ``new_weight = decay * old_weight + (1 - decay) * score``

    Parameters
    ----------
    (other arguments are explained in ``WeightScheme``)

    decay
        Decay parameter in [0, 1]. This parameter is used to weigh the
        running performance of each operator.
    """

    def __init__(
        self,
        scores: List[float],
        num_destroy: int,
        num_repair: int,
        decay: float,
        *,
        op_coupling: Optional[np.ndarray] = None
    ):
        super().__init__(
            scores, num_destroy, num_repair, op_coupling=op_coupling
        )

        if not (0 <= decay <= 1):
            raise ValueError("decay outside [0, 1] not understood.")

        self._decay = decay

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
        coupled_r_idcs = np.flatnonzero(self.op_coupling[d_idx])
        r_idx = coupled_r_idcs[select(self._r_weights[coupled_r_idcs])]

        return d_idx, r_idx

    def update(self, d_idx, r_idx, s_idx):
        self._d_weights[d_idx] *= self._decay
        self._d_weights[d_idx] += (1 - self._decay) * self._scores[s_idx]

        self._r_weights[r_idx] *= self._decay
        self._r_weights[r_idx] += (1 - self._decay) * self._scores[s_idx]
