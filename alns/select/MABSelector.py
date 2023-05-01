import itertools
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from mabwiser.mab import MAB
from mabwiser.utils import Num
from numpy.random import RandomState

from alns.State import State
from alns.select.OperatorSelectionScheme import OperatorSelectionScheme


class MABSelector(OperatorSelectionScheme):
    def __init__(
        self,
        mab: MAB,
        scores: List[float],
        num_destroy: int,
        num_repair: int,
        rnd_state: RandomState,
        context_extractor: Optional[
            Callable[[State], Union[List[Num], np.ndarray]]
        ] = None,
        op_coupling: Optional[np.ndarray] = None,
    ):
        super().__init__(num_destroy, num_repair, op_coupling)

        if any(score < 0 for score in scores):
            raise ValueError("Negative scores are not understood.")

        if len(scores) < 4:
            # More than four is OK because we only use the first four.
            raise ValueError(f"Expected four scores, found {len(scores)}")

        # the set of valid operator pairs is equal to the cartesian product
        # of destroy and repair operators
        options = [
            (d_idx, r_idx)
            for d_idx, r_idx in itertools.product(
                range(num_destroy), range(num_repair)
            )
            if self.op_coupling[d_idx, r_idx]
        ]

        self._mab = MAB(
            arms=[f"{i}_{j}" for (i, j) in options],
            learning_policy=mab.learning_policy,
            neighborhood_policy=mab.neighborhood_policy,
            seed=mab.seed,
            n_jobs=mab.n_jobs,
            backend=mab.backend,
        )
        self._scores = scores
        self._primed = False

        def extract_context(state):
            if context_extractor is None:
                return None
            else:
                context = context_extractor(state)
                if isinstance(context, list):
                    # if the output is a list we need to wrap it so it's 2D.
                    # The other case is that it's an np array or dataframe, in
                    # which case we leave it alone
                    context = [context]
                return context

        self._context_extractor = extract_context

    @property
    def scores(self) -> List[float]:
        return self._scores

    def __call__(
        self, rnd_state: RandomState, best: State, curr: State
    ) -> Tuple[int, int]:
        if not self._primed:
            # TODO: this is not valid if (0,0) is disallowed by op_coupling
            return (0, 0)
        else:
            prediction = self._mab.predict(
                contexts=self._context_extractor(curr)
            )
            # FIXME: what do we do if prediction is disallowed? Pick max from
            # predict_expectations?
            d_idx, r_idx = prediction.split("_")
            return int(d_idx), int(r_idx)

    def update(self, cand, d_idx, r_idx, outcome):
        if not self._primed:
            self._mab.fit(
                [f"{d_idx}_{r_idx}"],
                [self._scores[outcome]],
                contexts=self._context_extractor(cand),
            )
            self._primed = True
        else:
            self._mab.partial_fit(
                [f"{d_idx}_{r_idx}"],
                [self._scores[outcome]],
                contexts=self._context_extractor(cand),
            )
