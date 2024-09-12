from typing import List, Optional

import numpy as np

from alns.select.OperatorSelectionScheme import OperatorSelectionScheme


class AlphaUCB(OperatorSelectionScheme):
    """
    The :math:`\\alpha`-UCB (upper confidence bound) bandit scheme adapted
    from Hendel (2022).

    The action space :math:`A` is defined as each pair of (destroy, repair)
    operators that is allowed by the operator coupling matrix. The
    :math:`\\alpha`-UCB algorithm plays the following action in each iteration
    :math:`t`, computed as

    .. math::

        Q(t) = \\arg \\max_{a \\in A} \\left\\{ \\bar{r}_a (t - 1)
               + \\sqrt{\\frac{\\alpha \\ln(1 + t)}{T_a (t - 1)}} \\right\\},

    where :math:`T_a(t - 1)` is the number of times action :math:`a` has been
    played, and :math:`\\bar r_a(t - 1)` is the average reward of action
    :math:`a`, both in the first :math:`t - 1` iterations. The value that is
    maximised over the actions :math:`a \\in A` consists of the average reward
    term :math:`\\bar r_a(t - 1)` and an exploration bonus depending on
    :math:`t` and the number of times :math:`a` has been played.

    See :meth:`~alns.select.AlphaUCB.AlphaUCB.update` for details on how the
    average reward :math:`\\bar r_a` is updated.

    .. note::

        The average reward :math:`\\bar r_a(0)` of each action :math:`a \\in A`
        is initialised to 1. The ``scores`` list passed into the
        :math:`\\alpha`-UCB scheme should be 'reasonable' with respect to this
        default.

    Parameters
    ----------
    scores
        A list of four non-negative elements, representing the rewards when the
        candidate solution results in a new global best (idx 0), is better than
        the current solution (idx 1), the solution is accepted (idx 2), or
        rejected (idx 3).
    alpha
        The :math:`\\alpha \\in [0, 1]` parameter controls how much exploration
        is performed. Values of :math:`\\alpha`  near one result in much
        exploration, whereas values of :math:`\\alpha` nearer to zero result in
        more exploitation of good operator pairs. Typically,
        :math:`\\alpha \\le 0.1` is a good choice.
    num_destroy
        Number of destroy operators.
    num_repair
        Number of repair operators.
    op_coupling
        Optional boolean matrix that indicates coupling between destroy and
        repair operators. Entry (i, j) is True if destroy operator i can be
        used together with repair operator j, and False otherwise.

    References
    ----------
    .. [1] Hendel, G. 2022. Adaptive large neighborhood search for mixed
           integer programming. *Mathematical Programming Computation* 14:
           185 - 221.
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
            raise ValueError(f"Alpha {alpha:} outside [0, 1] not understood.")

        if any(score < 0 for score in scores):
            raise ValueError("Negative scores are not understood.")

        if len(scores) < 4:
            # More than four is OK because we only use the first four.
            raise ValueError(f"Expected four scores, found {len(scores)}")

        self._scores = scores
        self._alpha = alpha

        self._avg_rewards = np.ones_like(self._op_coupling, dtype=float)
        self._times = np.zeros_like(self._op_coupling, dtype=int)
        self._iter = 0  # current iteration

    @property
    def scores(self) -> List[float]:
        return self._scores

    @property
    def alpha(self) -> float:
        return self._alpha

    def __call__(self, rng, best, curr):
        """
        Returns the (destroy, repair) operator pair that maximises the average
        reward and exploration bonus.
        """
        action = np.argmax(self._values())
        return tuple(np.unravel_index(action, self.op_coupling.shape))

    def update(self, candidate, d_idx, r_idx, outcome):
        """
        Updates the average reward of the given destroy and repair operator
        combination ``(d_idx, r_idx)``.

        In particular, the reward of the action :math:`a` associated with this
        operator combination is updated as

        .. math::

            \\bar r_a (t) = \\frac{T_a(t - 1) \\bar r_a(t - 1)
                            + \\text{scores}[\\text{outcome}]}{T_a(t - 1) + 1},

        and :math:`T_a(t) = T_a(t - 1) + 1`.
        """
        # Update everything for the next iteration (t + 1)
        t_a = self._times[d_idx, r_idx]
        r = self._avg_rewards[d_idx, r_idx]
        avg_reward = (t_a * r + self.scores[outcome]) / (t_a + 1)

        self._avg_rewards[d_idx, r_idx] = avg_reward
        self._times[d_idx, r_idx] += 1
        self._iter += 1

    def _values(self):
        a = self.alpha
        t = self._iter

        value = self._avg_rewards
        explore_bonus = np.sqrt((a * np.log(1 + t)) / (self._times + 1))

        values = value + explore_bonus
        values[~self._op_coupling] = -1  # avoid selecting disallowed pairs

        return values
