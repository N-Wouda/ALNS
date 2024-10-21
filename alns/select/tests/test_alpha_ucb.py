from typing import List

import numpy as np
import numpy.random as rnd
from numpy.testing import assert_equal, assert_raises
from pytest import mark

from alns.Outcome import Outcome
from alns.select import AlphaUCB
from alns.tests.states import Zero


@mark.parametrize(
    "scores, alpha",
    [
        ([0, 0, 0, 0], 0),
        ([0, 1, 2, 3], 0.5),
        ([5, 3, 2, 1], 1),
    ],
)
def test_properties(scores, alpha):
    select = AlphaUCB(scores, alpha, 1, 1)

    assert_equal(select.scores, scores)
    assert_equal(select.alpha, alpha)


@mark.parametrize("alpha", [1.01, -0.01, -0.5, 1.05, 1.5])
def test_raises_invalid_alpha(alpha: float):
    with assert_raises(ValueError):
        AlphaUCB([0, 0, 0, 0], alpha, 1, 1)


@mark.parametrize("alpha", np.linspace(0, 1, num=5))
def test_does_not_raise_valid_decay(alpha: float):
    AlphaUCB([0, 0, 0, 0], alpha, 1, 1)


@mark.parametrize(
    "scores, alpha, num_destroy, num_repair",
    [
        ([5, 3, 2, -1], 0.5, 1, 1),  # negative score
        ([5, 3, 2], 0.5, 1, 1),  # len(score) < 4
        ([5, 3, 2, 1], -0.1, 1, 1),  # alpha < 0
        ([5, 3, 2, 1], 1.1, 1, 1),  # alpha > 1
    ],
)
def test_raises_invalid_arguments(
    scores: List[float],
    alpha: float,
    num_destroy: int,
    num_repair: int,
):
    with assert_raises(ValueError):
        AlphaUCB(scores, alpha, num_destroy, num_repair)


def test_call_with_only_one_operator_pair():
    # Only one operator pair, so the algorithm should select (0, 0).
    select = AlphaUCB([2, 1, 1, 0], 0.5, 1, 1)
    rng = rnd.default_rng()

    selected = select(rng, Zero(), Zero())
    assert_equal(selected, (0, 0))


def test_update_with_two_operator_pairs():
    select = AlphaUCB([2, 1, 1, 0], 0.5, 2, 1)
    rng = rnd.default_rng()

    # Avg. reward for (0, 0) after this is 2, for (1, 0) is still 1 (default).
    select.update(Zero(), 0, 0, outcome=Outcome.BEST)

    # So now (0, 0) is selected again.
    selected = select(rng, Zero(), Zero())
    assert_equal(selected, (0, 0))

    # One more update. Avg. reward goes to 1, and number of times to 2.
    select.update(Zero(), 0, 0, outcome=Outcome.REJECT)

    # The Q value of (0, 0) is now approx 1.432, and that of (1, 0) is now
    # approx 1.74. So (1, 0) is selected.
    selected = select(rng, Zero(), Zero())
    assert_equal(selected, (1, 0))
