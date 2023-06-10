from typing import List, Optional, Tuple

import numpy as np
from numpy.random import RandomState

from alns.Outcome import Outcome
from alns.State import ContextualState
from alns.select.OperatorSelectionScheme import OperatorSelectionScheme

try:
    from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

    MABWISER_AVAILABLE = True
except ModuleNotFoundError:
    MABWISER_AVAILABLE = False


class MABSelector(OperatorSelectionScheme):
    """
    A selector that uses any multi-armed-bandit algorithm from MABWiser.

    This selector is a wrapper around the many multi-armed bandit algorithms
    available in the `MABWiser <https://github.com/fidelity/mabwiser>`_
    library. Since ALNS operator selection can be framed as a
    multi-armed-bandit problem (where each [destroy, repair] operator pair is
    a bandit arm), this wrapper allows you to use a variety of existing
    multi-armed-bandit algorithms as operator selectors instead of
    having to reimplement them.

    Note that if the provided learning policy is a contextual bandit
    algorithm, your state class must provide a `get_context` function that
    returns a context vector for the current state.

    Parameters
    ----------
    scores
        A list of four non-negative elements, representing the rewards when the
        candidate solution results in a new global best (idx 0), is better than
        the current solution (idx 1), the solution is accepted (idx 2), or
        rejected (idx 3).
    num_destroy
        Number of destroy operators.
    num_repair
        Number of repair operators.
    learning_policy
        A MABWiser learning policy that acts as an operator selector. See the
        MABWiser documentation for a list of available learning policies.
    neighborhood_policy
        The neighborhood policy that MABWiser should use. Only available for
        contextual learning policies. See the MABWiser documentation for a
        list of available neighborhood policies.
    seed
        A seed that will be passed to the underlying MABWiser object.
    op_coupling
        Optional boolean matrix that indicates coupling between destroy and
        repair operators. Entry (i, j) is True if destroy operator i can be
        used together with repair operator j, and False otherwise.
    kwargs
        Any additional arguments. These will be passed to the underlying MAB
        object.

    References
    ----------
    .. [1] Emily Strong, Bernard Kleynhans, & Serdar Kadioglu (2021).
           MABWiser: Parallelizable Contextual Multi-armed Bandits.
           Int. J. Artif. Intell. Tools, 30(4), 2150021: 1-19.
    """

    def __init__(
        self,
        scores: List[float],
        num_destroy: int,
        num_repair: int,
        learning_policy: "LearningPolicy",
        neighborhood_policy: Optional["NeighborhoodPolicy"] = None,
        seed: Optional[int] = None,
        op_coupling: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if not MABWISER_AVAILABLE:
            raise ImportError("MABSelector requires the MABWiser library. ")

        super().__init__(num_destroy, num_repair, op_coupling)

        if any(score < 0 for score in scores):
            raise ValueError("Negative scores are not understood.")

        if len(scores) < 4:
            # More than four is OK because we only use the first four.
            raise ValueError(f"Expected four scores, found {len(scores)}")

        self._scores = scores

        if seed is not None:
            kwargs["seed"] = seed

        arms = [
            f"{d_idx}_{r_idx}"
            for d_idx in range(num_destroy)
            for r_idx in range(num_repair)
            if self._op_coupling[d_idx, r_idx]
        ]

        self._mab = MAB(arms, learning_policy, neighborhood_policy, **kwargs)

    @property
    def scores(self) -> List[float]:
        return self._scores

    @property
    def mab(self) -> "MAB":
        return self._mab

    def __call__(  # type: ignore[override]
        self,
        rnd_state: RandomState,
        best: ContextualState,
        curr: ContextualState,
    ) -> Tuple[int, int]:
        """
        Returns the (destroy, repair) operator pair from the underlying MAB
        strategy
        """
        if self._mab._is_initial_fit:
            has_ctx = self._mab.is_contextual
            ctx = np.atleast_2d(curr.get_context()) if has_ctx else None
            prediction = self._mab.predict(contexts=ctx)
            return arm2ops(prediction)
        else:
            # This can happen when the MAB object has not yet been fit on any
            # observations. In that case we return any feasible operator index
            # pair as a first observation.
            allowed = np.argwhere(self._op_coupling)
            idx = rnd_state.randint(len(allowed))
            return allowed[idx][0], allowed[idx][1]

    def update(  # type: ignore[override]
        self,
        cand: ContextualState,
        d_idx: int,
        r_idx: int,
        outcome: Outcome,
    ):
        """
        Updates the underlying MAB algorithm given the reward of the chosen
        destroy and repair operator combination ``(d_idx, r_idx)``.
        """
        has_ctx = self._mab.is_contextual
        ctx = np.atleast_2d(cand.get_context()) if has_ctx else None
        self._mab.partial_fit(
            [ops2arm(d_idx, r_idx)],
            [self._scores[outcome]],
            contexts=ctx,
        )


def ops2arm(d_idx: int, r_idx: int) -> str:
    """
    Converts the given destroy and repair operator indices to an arm string
    that can be passed to the MAB instance.

    Examples
    --------
    >>> ops2arm(0, 1)
    "0_1"
    >>> ops2arm(12, 3)
    "12_3"
    """
    return f"{d_idx}_{r_idx}"


def arm2ops(arm: str) -> Tuple[int, int]:
    """
    Converts an arm string returned by the MAB instance into a tuple of destroy
    and repair operator indices.

    Examples
    --------
    >>> arm2ops("0_1")
    (0, 1)
    >>> arm2ops("12_3")
    (12, 3)
    """
    d_idx, r_idx = map(int, arm.split("_"))
    return d_idx, r_idx
