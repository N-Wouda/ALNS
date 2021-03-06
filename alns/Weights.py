import numpy as np


class Weights:

    def __init__(self, num_destroy, num_repair):
        """
        TODO
        """
        self._d_weights = np.ones(num_destroy, dtype=np.float16)
        self._r_weights = np.ones(num_repair, dtype=np.float16)

    @property
    def d_weights(self):
        return self._d_weights

    @property
    def r_weights(self):
        return self._r_weights

    def update_destroy(self, d_idx, op_decay, weight):
        self._d_weights[d_idx] *= op_decay
        self._d_weights[d_idx] += (1 - op_decay) * weight

    def update_repair(self, r_idx, op_decay, weight):
        self._r_weights[r_idx] *= op_decay
        self._r_weights[r_idx] += (1 - op_decay) * weight

    def reset_weights(self):
        self._d_weights = np.ones_like(self.d_weights)
        self._r_weights = np.ones_like(self.r_weights)
