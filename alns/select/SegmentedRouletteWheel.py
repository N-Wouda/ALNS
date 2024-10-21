import logging
from typing import List, Optional

import numpy as np

from alns.State import State
from alns.select.RouletteWheel import RouletteWheel

logger = logging.getLogger(__name__)


class SegmentedRouletteWheel(RouletteWheel):
    R"""
    .. note::

        First read the documentation for
        :class:`~alns.select.RouletteWheel.RouletteWheel`, the parent of this
        class.

    The ``SegmentedRouletteWheel`` scheme extends the
    :class:`~alns.select.RouletteWheel.RouletteWheel` scheme by fixing operator
    weights for a number of iterations (the *segment length*). This allows
    certain sets of operators to be selected more often in different
    neighbourhoods.

    Initially, all weights are set to one, as in
    :class:`~alns.select.RouletteWheel.RouletteWheel`.
    A separate score is tracked for each operator :math:`d` and :math:`r`, to
    which the observed scores :math:`s_j` are added in each iteration where
    :math:`d` and :math:`r` are applied. After the segment concludes, these
    summed scores are added to the existing weights :math:`\omega_d` and
    :math:`\omega_r` as a convex combination using a parameter :math:`\theta`
    (the *segment decay rate*), as in ``RouletteWheel``. The separate
    score list is then reset to zero, and a new segment begins.

    Parameters
    ----------
    scores
        A list of four non-negative elements, representing the weight
        updates when the candidate solution results in a new global best
        (idx 0), is better than the current solution (idx 1), the solution
        is accepted (idx 2), or rejected (idx 3).
    decay
        Operator decay parameter :math:`\theta \in [0, 1]`. This parameter is
        used to weigh the running performance of each operator.
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

    def __call__(self, rng, best: State, curr: State):
        self._iter += 1

        if self._iter % self._seg_length == 0:
            logger.debug(f"End of segment (#iters = {self._iter}).")

            self._d_weights *= self._decay
            self._d_weights += (1 - self._decay) * self._d_seg_weights

            self._r_weights *= self._decay
            self._r_weights += (1 - self._decay) * self._r_seg_weights

            self._reset_segment_weights()

        return super().__call__(rng, best, curr)

    def update(self, cand, d_idx, r_idx, outcome):
        self._d_seg_weights[d_idx] += self._scores[outcome]
        self._r_seg_weights[r_idx] += self._scores[outcome]

    def _reset_segment_weights(self):
        self._d_seg_weights = np.zeros_like(self._d_weights)
        self._r_seg_weights = np.zeros_like(self._r_weights)
