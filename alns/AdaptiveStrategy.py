class AdaptiveStrategy:

    # TODO weight lists for both operator lists.

    def __init__(self,
                 init_weights,
                 update_weights,
                 reset_every,
                 combine_rule):
        """
        TODO
        """
        self._init_weights = init_weights
        self._update_weights = update_weights

        self._weights = init_weights

        self._reset_every = reset_every
        self.combine = combine_rule
        self._iteration = 0

    def current_weights(self):
        """
        TODO
        """
        return self._weights

    def update_weights(self, operator_idx, weight_idx):
        """
        TODO
        """
        self._iteration += 1

        self._weights[operator_idx] = self.combine(
            self._weights[operator_idx],
            self._update_weights[weight_idx])

        if self._iteration % self._reset_every == 0:
            self.reset_weights()

    def reset_weights(self):
        """
        TODO
        """
        self._weights = self._init_weights
