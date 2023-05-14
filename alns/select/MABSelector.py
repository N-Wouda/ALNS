import itertools
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from mabwiser.mab import MAB
from mabwiser.utils import Num
from numpy.random import RandomState

from alns.State import State
from alns.select.OperatorSelectionScheme import OperatorSelectionScheme


class MABSelector(OperatorSelectionScheme):
    """
    A selector that uses any multi-armed-bandit algorithm from MABWiser.

    This selector is a wrapper around the many multi-armed bandit algorithms
    available in the [MABWiser](https://github.com/fidelity/mabwiser) library.
    Since ALNS operator selection can be framed as a multi-armed-bandit
    problem, this wrapper allows you to use a variety of existing
    multi-armed-bandit algorithms as operator selectors instead of having to
    reimplement them.

    Parameters
    ----------
    scores
        A list of four non-negative elements, representing the rewards when the
        candidate solution results in a new global best (idx 0), is better than
        the current solution (idx 1), the solution is accepted (idx 2), or
        rejected (idx 3).
    mab
        A mabwiser MAB object that will be used to select the
        (destroy, repair) operator pairs.
    num_destroy
        Number of destroy operators.
    num_repair
        Number of repair operators.
    op_coupling
        Optional boolean matrix that indicates coupling between destroy and
        repair operators. Entry (i, j) is True if destroy operator i can be
        used together with repair operator j, and False otherwise.
    context_extractor
        Optional function that takes a State object and returns a context
        vector for that state that can be passed to a contextual mabwiser
        bandit. If the MAB algorithm supports it, this context will be used to
        help predict the next (destroy, repair) combination.

    References
    ----------
    .. [1] Emily Strong, Bernard Kleynhans, & Serdar Kadioglu (2021).
           MABWiser: Parallelizable Contextual Multi-armed Bandits.
           Int. J. Artif. Intell. Tools, 30(4), 2150021:1–2150021:19.
    """

    @staticmethod
    def make_arms(
        num_destroy: int,
        num_repair: int,
        op_coupling: Optional[np.ndarray] = None,
    ) -> List[str]:
        """
        Generates a list of arms for the MAB passed to MABSelector.

        Any MABs passed to MABSelector must be generated with this function,
        and with the same `num_destroy`, `num_repair`, and `op_coupling`
        parameters.
        """
        # the set of valid operator pairs is equal to the cartesian product
        # of destroy and repair operators, except we leave out any pairs
        # disallowed by op_coupling
        return [
            f"{d_idx}_{r_idx}"
            for d_idx, r_idx in itertools.product(
                range(num_destroy), range(num_repair)
            )
            if op_coupling is None or op_coupling[d_idx, r_idx]
        ]

    def __init__(
        self,
        scores: List[float],
        mab: MAB,
        num_destroy: int,
        num_repair: int,
        op_coupling: Optional[np.ndarray] = None,
        context_extractor: Optional[
            Callable[
                [State], Union[List[Num], np.ndarray, pd.Series, pd.DataFrame]
            ]
        ] = None,
    ):
        super().__init__(num_destroy, num_repair, op_coupling)

        if any(score < 0 for score in scores):
            raise ValueError("Negative scores are not understood.")

        if len(scores) < 4:
            # More than four is OK because we only use the first four.
            raise ValueError(f"Expected four scores, found {len(scores)}")

        if mab.arms != self.make_arms(num_destroy, num_repair, op_coupling):
            raise ValueError(
                "Arms in MAB passed to MABSelector are incorrect. Make sure"
                + "to generate arms with `MABSelector.make_arms`"
            )

        self._mab = mab
        self._scores = scores
        self._primed = False
        self._op_coupling = op_coupling

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

    @staticmethod
    def _operators_to_arm(destroy_idx: int, repair_idx: int) -> str:
        """
        Converts a tuple of destroy and repair operator indices to an arm
        string that can be passed to self._mab.

        Examples
        --------
        >>> MABSelector._operators_to_arm(0, 1)
        "0_1"
        >>> MABSelector._operators_to_arm(12, 3)
        "12_3"
        """
        return f"{destroy_idx}_{repair_idx}"

    @staticmethod
    def _arm_to_operators(arm: str) -> Tuple[int, int]:
        """
        Converts an arm string returned from self._mab to a tuple of destroy
        and repair operator indices.

        Examples
        --------
        >>> MABSelector._arm_to_operators("0_1")
        (0, 1)
        >>> MABSelector._arm_to_operators("12_3")
        (12, 3)
        """
        [destroy, repair] = arm.split("_")
        return int(destroy), int(repair)

    def __call__(
        self, rnd_state: RandomState, best: State, curr: State
    ) -> Tuple[int, int]:
        """
        Returns the (destroy, repair) operator pair from the underlying MAB
        strategy
        """
        if not self._primed:
            # Default: return the first allowed operator pair
            if self._op_coupling is None:
                return 0, 0
            else:
                first_non_zero = np.argmax(self._op_coupling)
                as_indices = np.unravel_index(
                    first_non_zero, self._op_coupling.shape
                )
                return as_indices
        else:
            prediction = self._mab.predict(
                contexts=self._context_extractor(curr)
            )
            return self._arm_to_operators(prediction)

    def update(self, cand, d_idx, r_idx, outcome):
        """
        Updates the underlying MAB algorithm given the reward of the chosen
        destroy and repair operator combination `(d_idx, r_idx)`.
        """
        if not self._primed:
            self._mab.fit(
                [self._operators_to_arm(d_idx, r_idx)],
                [self._scores[outcome]],
                contexts=self._context_extractor(cand),
            )
            self._primed = True
        else:
            self._mab.partial_fit(
                [self._operators_to_arm(d_idx, r_idx)],
                [self._scores[outcome]],
                contexts=self._context_extractor(cand),
            )
