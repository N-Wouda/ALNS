import numpy as np

from alns.select.OperatorSelectionScheme import OperatorSelectionScheme


class RandomSelect(OperatorSelectionScheme):
    """
    Randomly selects operator pairs with uniform probability. The operator
    pairs respect the operator coupling matrix.
    """

    def __call__(self, rnd, best, curr):
        """
        Selects a destroy operator with uniform probability. Then selects a
        repair operator with uniform probability, respecting the operator
        coupling matrix.
        """
        d_idx = rnd.randint(0, self.num_destroy)

        r_indices = np.flatnonzero(self.op_coupling[d_idx, :])
        r_idx = rnd.choice(r_indices)

        return d_idx, r_idx

    def update(self, candidate, d_idx, r_idx, s_idx):
        pass
