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

    def select_destroy(self, rnd_state):
        return rnd_state.choice(np.arange(0, len(self.d_weights)),
                                p=self.d_weights / np.sum(self.d_weights))

    def update_destroy(self, d_idx, op_decay, weight):
        self._d_weights[d_idx] *= op_decay
        self._d_weights[d_idx] += (1 - op_decay) * weight

    def select_repair(self, rnd_state):
        return rnd_state.choice(np.arange(0, len(self.r_weights)),
                                p=self.r_weights / np.sum(self.r_weights))

    def update_repair(self, r_idx, op_decay, weight):
        self._r_weights[r_idx] *= op_decay
        self._r_weights[r_idx] += (1 - op_decay) * weight

    def reset_weights(self):
        self._d_weights = np.ones_like(self.d_weights)
        self._r_weights = np.ones_like(self.r_weights)

    def _update_weights(self):
        """
        The weights are updated as convex combinations of the current weight and
        the update parameter. See eq. (2), p. 12. TODO
        """
        pass

    def _select_operator(self):
        pass
