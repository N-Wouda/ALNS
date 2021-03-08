from typing import List

import numpy as np

from .WeightScheme import WeightScheme


class ConvexWeights(WeightScheme):

    def __init__(self,
                 op_decay: float,
                 weights: List[float],
                 num_destroy: int,
                 num_repair: int):
        """
        TODO
        """
        super().__init__(weights, num_destroy, num_repair)

        if not (0 <= op_decay <= 1):
            raise ValueError("Operator decay outside [0, 1]] not understood.")

        self._op_decay = op_decay

    def select_operators(self, rnd_state):
        def select(op_weights):
            probs = op_weights / np.sum(op_weights)
            return rnd_state.choice(range(len(op_weights)), p=probs)

        return select(self._d_weights), select(self._r_weights)

    def update_weights(self, d_idx, r_idx, w_idx):
        self._d_weights[d_idx] *= self._op_decay
        self._d_weights[d_idx] += (1 - self._op_decay) * self._weights[w_idx]

        self._r_weights[r_idx] *= self._op_decay
        self._r_weights[r_idx] += (1 - self._op_decay) * self._weights[w_idx]
