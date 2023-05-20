import itertools
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState

from alns.State import State
from alns.select.OperatorSelectionScheme import OperatorSelectionScheme

MABWISER_AVAILABLE = True
try:
    from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
    from mabwiser.utils import Num
except ModuleNotFoundError:
    MABWISER_AVAILABLE = False


def ops2arm(destroy_idx: int, repair_idx: int) -> str:
    """
    Converts a tuple of destroy and repair operator indices to an arm
    string that can be passed to self._mab.

    Examples
    --------
    >>> ops2arm(0, 1)
    "0_1"
    >>> ops2arm(12, 3)
    "12_3"
    """
    return f"{destroy_idx}_{repair_idx}"


def arm2ops(arm: str) -> Tuple[int, int]:
    """
    Converts an arm string returned from self._mab to a tuple of destroy
    and repair operator indices.

    Examples
    --------
    >>> arm2ops("0_1")
    (0, 1)
    >>> arm2ops("12_3")
    (12, 3)
    """
    [destroy, repair] = arm.split("_")
    return int(destroy), int(repair)


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

    Note that the supplied ``MAB`` object must be generated with the static
    method ``make_arms``.

    Parameters
    ----------
    scores
        A list of four non-negative elements, representing the rewards when the
        candidate solution results in a new global best (idx 0), is better than
        the current solution (idx 1), the solution is accepted (idx 2), or
        rejected (idx 3).
    mab
        A MABWiser MAB object that will be used to select the
        (destroy, repair) operator pairs. The arms of the ``mab`` object must
        be generated with the static method ``make_arms``.
    num_destroy
        Number of destroy operators.
    num_repair
        Number of repair operators.
    op_coupling
        Optional boolean matrix that indicates coupling between destroy and
        repair operators. Entry (i, j) is True if destroy operator i can be
        used together with repair operator j, and False otherwise.
    context_extractor
        Optional function that takes a ALNS State object and returns a context
        vector for that state that can be passed to a contextual MABWiser
        bandit. If the MAB algorithm supports it, this context will be used to
        help predict the next (destroy, repair) combination.

    References
    ----------
    .. [1] Emily Strong, Bernard Kleynhans, & Serdar Kadioglu (2021).
           MABWiser: Parallelizable Contextual Multi-armed Bandits.
           Int. J. Artif. Intell. Tools, 30(4), 2150021:1–2150021:19.
    """

    if not MABWISER_AVAILABLE:
        raise ImportError("MABSelector requires the MABWiser library. ")

    def __init__(
        self,
        scores: List[float],
        num_destroy: int,
        num_repair: int,
        learning_policy: LearningPolicy,
        neighborhood_policy: Optional[NeighborhoodPolicy] = None,
        seed: Optional[int] = None,
        op_coupling: Optional[np.ndarray] = None,
        context_extractor: Optional[
            Callable[
                [State], Union[List[Num], np.ndarray, pd.Series, pd.DataFrame]
            ]
        ] = None,
        **kwargs,
    ):
        super().__init__(num_destroy, num_repair, op_coupling)

        if any(score < 0 for score in scores):
            raise ValueError("Negative scores are not understood.")

        if len(scores) < 4:
            # More than four is OK because we only use the first four.
            raise ValueError(f"Expected four scores, found {len(scores)}")

        # forward the seed argument if not null
        if seed is not None:
            kwargs["seed"] = seed

        # the set of valid operator pairs (arms) is equal to the cartesian
        # product of destroy and repair operators, except we leave out any
        # pairs disallowed by op_coupling
        arms = [
            f"{d_idx}_{r_idx}"
            for d_idx, r_idx in itertools.product(
                range(num_destroy), range(num_repair)
            )
            if op_coupling is None or op_coupling[d_idx, r_idx]
        ]
        self._mab = MAB(
            arms,
            learning_policy,
            neighborhood_policy,
            **kwargs,
        )
        self._scores = scores

        def extract_context(state):
            if context_extractor is None:
                return None
            else:
                context = context_extractor(state)
                if isinstance(context, list):
                    # if the output is a list, wrap it so it's 2D. Otherwise,
                    # it's an np array or dataframe, which can be left alone.
                    context = [context]
                return context

        self._context_extractor = extract_context

    @property
    def scores(self) -> List[float]:
        return self._scores

    @property
    def mab(self) -> MAB:
        return self._mab

    def __call__(
        self, rnd_state: RandomState, best: State, curr: State
    ) -> Tuple[int, int]:
        """
        Returns the (destroy, repair) operator pair from the underlying MAB
        strategy
        """
        try:
            prediction = self._mab.predict(
                contexts=self._context_extractor(curr)
            )
            return arm2ops(prediction)
        except Exception:
            # This can happen when the MAB object has not yet been fit on any
            # observations. In that case we return any feasible operator index
            # pair as a first observation.
            allowed = np.argwhere(self._op_coupling)
            idx = rnd_state.randint(len(allowed))
            return (allowed[idx][0], allowed[idx][1])

    def update(self, cand, d_idx, r_idx, outcome):
        """
        Updates the underlying MAB algorithm given the reward of the chosen
        destroy and repair operator combination ``(d_idx, r_idx)``.
        """
        self._mab.partial_fit(
            [ops2arm(d_idx, r_idx)],
            [self._scores[outcome]],
            contexts=self._context_extractor(cand),
        )
