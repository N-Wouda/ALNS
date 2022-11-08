import numpy as np

from alns.select.OperatorSelectionScheme import OperatorSelectionScheme


class RandomSelect(OperatorSelectionScheme):
    """
    Randomly selects operator pairs with uniform probability. The operator
    pairs respect the operator coupling matrix.
    """

    def __call__(self, rnd, best, curr):
        """
        Selects a (destroy, repair) operator pair with uniform probability.
        """
        allowed = np.argwhere(self._op_coupling)
        idx = rnd.randint(len(allowed))

        return tuple(allowed[idx])

    def update(self, candidate, d_idx, r_idx, outcome):
        pass  # pragma: no cover
