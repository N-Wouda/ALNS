from typing import List

import numpy as np

from .WeightScheme import WeightScheme


class SegmentedWeights(WeightScheme):

    def __init__(self,
                 scores: List[float],
                 num_destroy: int,
                 num_repair: int,
                 seg_decay: float,
                 seg_length: int = 100):
        """
        A segmented weight scheme. Weights are not updated in each iteration,
        but only after each segment. Scores are gathered during each segment,
        as:

        ``seg_weight += score``

        At the start of each segment, ``seg_weight`` is reset to zero. At the
        end of a segment, the weights are updated as:

        ``new_weight = seg_decay * old_weight + (1 - seg_decay) * seg_weight``

        Parameters
        ----------
        (other arguments are explained in ``WeightScheme``)

        seg_decay
            Decay parameter in [0, 1]. This parameter is used to weigh segment
            and overall performance of each operator.
        seg_length
            Length of a single segment. Default 100.
        """
        super().__init__(scores, num_destroy, num_repair)

        if not (0 <= seg_decay <= 1):
            raise ValueError("seg_decay outside [0, 1] not understood.")

        self._seg_decay = seg_decay
        self._seg_length = seg_length

        self._reset_segment_weights()

    def at_iteration_start(self, iteration, max_iterations):
        if iteration % self._seg_length == 0:
            self._d_weights *= self._seg_decay
            self._d_weights += (1 - self._seg_decay) * self._d_seg_weights

            self._r_weights *= self._seg_decay
            self._r_weights += (1 - self._seg_decay) * self._r_seg_weights

            self._reset_segment_weights()

    def update_weights(self, d_idx, r_idx, s_idx):
        self._d_seg_weights[d_idx] += self._scores[s_idx]
        self._r_seg_weights[r_idx] += self._scores[s_idx]

    def _reset_segment_weights(self):
        self._d_seg_weights = np.zeros_like(self._d_weights)
        self._r_seg_weights = np.zeros_like(self._r_weights)