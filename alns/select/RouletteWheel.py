from typing import List, Optional, Tuple

import numpy as np
from numpy.random import Generator

from alns.State import State
from alns.select.OperatorSelectionScheme import OperatorSelectionScheme


class RouletteWheel(OperatorSelectionScheme):
    R"""
    The ``RouletteWheel`` scheme updates operator weights as a convex
    combination of the current weight, and the new score.

    When the algorithm starts, all operators :math:`i` are assigned weight
    :math:`\omega_i = 1`. In each iteration, a destroy and a repair operator
    are selected by the ALNS algorithm, based on the normalised current weights
    :math:`\omega_i`. The selected operators are applied to the current
    solution, resulting in a new candidate solution. This candidate is
    evaluated by the ALNS algorithm, which leads to one of four outcomes:

    1. The candidate solution is a new global best.
    2. The candidate solution is better than the current solution, but not a
       new global best.
    3. The candidate solution is accepted.
    4. The candidate solution is rejected.

    Each of these four outcomes is assigned a score :math:`s_j` (with
    :math:`j = 1,...,4`). After observing outcome :math:`j`, the weights of
    the selected destroy and repair operators :math:`d` and :math:`r` that
    were applied are updated as follows:

    .. math::

        \begin{align}
            \omega_d &= \theta \omega_d + (1 - \theta) s_j, \\
            \omega_r &= \theta \omega_r + (1 - \theta) s_j,
        \end{align}

    where :math:`0 \le \theta \le 1` (known as the *operator decay rate*)
    is a parameter.

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
        num_destroy: int,
        num_repair: int,
        op_coupling: Optional[np.ndarray] = None,
    ):
        super().__init__(num_destroy, num_repair, op_coupling)

        if any(score < 0 for score in scores):
            raise ValueError("Negative scores are not understood.")

        if len(scores) < 4:
            # More than four is OK because we only use the first four.
            raise ValueError(f"Expected four scores, found {len(scores)}")

        if not (0 <= decay <= 1):
            raise ValueError("decay outside [0, 1] not understood.")

        self._scores = scores
        self._d_weights = np.ones(num_destroy, dtype=float)
        self._r_weights = np.ones(num_repair, dtype=float)
        self._decay = decay

    @property
    def scores(self) -> List[float]:
        return self._scores

    @property
    def destroy_weights(self) -> np.ndarray:
        return self._d_weights

    @property
    def repair_weights(self) -> np.ndarray:
        return self._r_weights

    @property
    def decay(self) -> float:
        return self._decay

    def __call__(
        self, rng: Generator, best: State, curr: State
    ) -> Tuple[int, int]:
        """
        Selects a destroy and repair operator pair to apply in this iteration.
        The probability of an operator being selected is based on the operator
        weights: operators that frequently improve the current solution - and
        thus have higher weights - are selected with a higher probability.

        Parameters
        ----------
        rng
            Random number generator.
        best
            The best solution state observed so far.
        current
            The current solution state.

        Returns
        -------
        A tuple of (d_idx, r_idx), which are indices into the destroy and
        repair operator lists, respectively.
        """

        def select(op_weights):
            probs = op_weights / np.sum(op_weights)
            return rng.choice(range(len(op_weights)), p=probs)

        d_idx = select(self._d_weights)
        coupled_r_idcs = np.flatnonzero(self.op_coupling[d_idx])
        r_idx = coupled_r_idcs[select(self._r_weights[coupled_r_idcs])]

        return d_idx, r_idx

    def update(self, cand, d_idx, r_idx, outcome):
        self._d_weights[d_idx] *= self._decay
        self._d_weights[d_idx] += (1 - self._decay) * self._scores[outcome]

        self._r_weights[r_idx] *= self._decay
        self._r_weights[r_idx] += (1 - self._decay) * self._scores[outcome]
