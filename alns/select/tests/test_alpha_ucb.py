from typing import List

import numpy as np
import numpy.random as rnd
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)
from pytest import mark

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


def test_update():
    pass


def test_call():
    pass
