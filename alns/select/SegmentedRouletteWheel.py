import logging
from typing import List, Optional

import numpy as np

from alns.select.RouletteWheel import RouletteWheel
from alns.State import State

logger = logging.getLogger(__name__)


class SegmentedRouletteWheel(RouletteWheel):
    """
    An operator selection scheme based on the roulette wheel mechanism, where
    each operator is selected based on normalised weights. Weights are not
    updated in each iteration, but only after each segment. Scores are gathered
    during each segment, as:

    ``seg_weight += score``

    At the start of each segment, ``seg_weight`` is reset to zero. At the end
    of a segment, the weights are updated as:

    ``new_weight = decay * old_weight + (1 - decay) * seg_weight``

    Parameters
    ----------
    scores
        A list of four non-negative elements, representing the weight
        updates when the candidate solution results in a new global best
        (idx 0), is better than the current solution (idx 1), the solution
        is accepted (idx 2), or rejected (idx 3).
    decay
        Decay parameter in [0, 1]. This parameter is used to weigh the
        running performance of each operator.
    seg_length
        Length of a single segment.
    num_destroy
        Number of destroy operators.
    num_repair
        Number of repair operators.
    op_coupling
        Optional boolean matrix that indicates coupling between destroy and
        repair operators. Entry (i, j) is True if destroy operator i can be
        used together with repair operator j, and False otherwise.
    """

    def __init__(
        self,
        scores: List[float],
        decay: float,
        seg_length: int,
        num_destroy: int,
        num_repair: int,
        op_coupling: Optional[np.ndarray] = None,
    ):
        super().__init__(scores, decay, num_destroy, num_repair, op_coupling)

        if seg_length < 1:
            raise ValueError("seg_length < 1 not understood.")

        self._seg_length = seg_length
        self._iter = 0

        self._reset_segment_weights()

    @property
    def seg_length(self):
        return self._seg_length

    def __call__(self, rnd_state, best: State, curr: State):
        self._iter += 1

        if self._iter % self._seg_length == 0:
            logger.debug(f"End of segment (#iters = {self._iter}).")

            self._d_weights *= self._decay
            self._d_weights += (1 - self._decay) * self._d_seg_weights

            self._r_weights *= self._decay
            self._r_weights += (1 - self._decay) * self._r_seg_weights

            self._reset_segment_weights()

        return super().__call__(rnd_state, best, curr)

    def update(self, cand, d_idx, r_idx, outcome):
        self._d_seg_weights[d_idx] += self._scores[outcome]
        self._r_seg_weights[r_idx] += self._scores[outcome]

    def _reset_segment_weights(self):
        self._d_seg_weights = np.zeros_like(self._d_weights)
        self._r_seg_weights = np.zeros_like(self._r_weights)
