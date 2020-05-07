class AdaptiveStrategy:

    # TODO weight lists for both operator lists.

    def __init__(self, init_weights, update_weights):
        """
        TODO
        """
        self._init_weights = init_weights
        self._update_weights = update_weights

        self._weights = init_weights

    def current_weights(self):
        """
        TODO
        """
        return self._weights

    def update_weights(self, operator_idx, weight_idx):
        """
        TODO
        """
        # TODO How should these be updated? Flexible strategy?
        pass

    def reset_weights(self):
        """
        TODO
        """
        self._weights = self._init_weights
