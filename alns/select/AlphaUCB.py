from typing import List, Optional

import numpy as np

from alns.select.OperatorSelectionScheme import OperatorSelectionScheme


class AlphaUCB(OperatorSelectionScheme):
    """
    The :math:`\\alpha`-UCB (upper confidence bound) bandit scheme of Hendel
    (2022).

    The action space :math:`A` is defined as each pair of (destroy, repair)
    operator that is allowed by the operator coupling matrix. The
    :math:`\\alpha`-UCB algorithm maintains a value for each action, computed as

    .. math::

        Q_{t} = \\arg \\max_{a \\in A} \\left\\{ \\bar{r}_a (t - 1)
                + \\sqrt{\\frac{\\alpha \\ln(1 + t)}{T_a (t - 1)}} \\right \\},

    where :math:`T_a(t - 1)` is the number of times action :math:`a` has been
    played, and :math:`\\bar r_a(t - 1)` is the average reward of action
    :math:`a`, both in the first :math:`t - 1` iterations.

    Initially, each action pair is played once. After that, the action
    :math:`Q(t)` is played for each later iteration :math:`t`. See
    :meth:`~alns.select.AlphaUCB.AlphaUCB.update` for details on how
    :math:`\\bar r_a` is updated.

    Parameters
    ----------
    scores
        A list of four non-negative elements, representing the rewards when the
        candidate solution results in a new global best (idx 0), is better than
        the current solution (idx 1), the solution is accepted (idx 2), or
        rejected (idx 3).
    alpha
        The :math:`\\alpha` parameter controls the width of the confidence
        interval. Larger values force the algorithm to select inferior
        operators more frequently, resulting in more exploration.
        :math:`\\alpha` must be in [0, 1].
    num_destroy
        Number of destroy operators.
    num_repair
        Number of repair operators.
    op_coupling
        Optional matrix that indicates coupling between destroy and repair
        operators. Entry (i, j) is 1 if destroy operator i can be used in
        conjunction with repair operator j and 0 otherwise.

    References
    ----------
    .. [1] Hendel, G. 2022. Adaptive large neighborhood search for mixed
           integer programming. *Mathematical Programming Computation* 14:
           185 – 221.
    """

    def __init__(
            self,
            scores: List[float],
            alpha: float,
            num_destroy: int,
            num_repair: int,
            op_coupling: Optional[np.ndarray] = None,
    ):
        super().__init__(num_destroy, num_repair, op_coupling)

        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha outside [0, 1] not understood.")

        if any(score < 0 for score in scores):
            raise ValueError("Negative scores are not understood.")

        if len(scores) < 4:
            # More than four is OK because we only use the first four.
            raise ValueError(f"Expected four scores, found {len(scores)}")

        self._scores = scores
        self._alpha = alpha

        # These are, in order, the number of times each pair has been played,
        # the average reward of each operator pair, the matrix of Q-values, and
        # the current iteration counter.
        self._times = np.zeros_like(self._op_coupling)
        self._rewards = np.zeros_like(self._op_coupling)

        # Set +inf for all operator pairs, except those that can never be
        # played. Those are set to zero.
        self._Q_values = np.full_like(self._op_coupling, np.inf)
        self._Q_values[~self._op_coupling] = 0

        self._iter = 0

    @property
    def scores(self) -> List[float]:
        return self._scores

    @property
    def alpha(self) -> float:
        return self._alpha

    def __call__(self, rnd, best, curr):
        """
        Returns the (destroy, repair) operator pair that maximises the average
        reward and exploration bonus.
        """
        return tuple(np.argmax(self._Q_values))

    def update(self, candidate, d_idx, r_idx, s_idx):
        """
        Updates the average reward of the given destroy and repair operator
        combination ``(d_idx, r_idx)``.

        In particular, the reward of the action :math:`a` associated with this
        operator combination is updated as

        .. math::

            \\bar r_a (t) = \\frac{T_a(t - 1) \\bar r_a(t - 1)
                            + \\text{scores}[\\text{s_idx}]}{T_a(t - 1) + 1},

        and :math:`T_a(t) = T_a (t - 1) + 1`.
        """
        # Update current iteration (that we are already in, so after this,
        # ``_iter`` corresponds to time t)
        self._iter += 1

        # Update everything for the next iteration (t + 1)
        a = self._alpha
        t = self._iter + 1
        t_a = self._times[d_idx, r_idx]
        r = self._rewards[d_idx, r_idx]

        avg_reward = (t_a * r + self.scores[s_idx]) / (t_a + 1)
        value = avg_reward + np.sqrt((a * np.log(1 + t)) / (t_a + 1))

        self._times[d_idx, r_idx] += 1
        self._rewards[d_idx, r_idx] = avg_reward
        self._Q_values[d_idx, r_idx] = value
