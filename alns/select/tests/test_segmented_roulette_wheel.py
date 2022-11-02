from typing import List

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
from pytest import mark

from alns.select import SegmentedRouletteWheel
from alns.tests.states import Zero


@mark.parametrize(
    "scores, num_destroy, num_repair, decay, seg_length,op_coupling",
    [
        ([0, 0, 0, 0], 1, 1, 0, 1, np.ones((1, 1))),
        ([0, 1, 2, 3], 2, 2, 0.2, 2, np.ones((2, 2))),
        ([5, 3, 2, 1], 10, 10, 1, 10, np.ones((10, 10))),
    ],
)
def test_properties(
    scores, num_destroy, num_repair, decay, seg_length, op_coupling
):
    weights = SegmentedRouletteWheel(
        scores,
        num_destroy,
        num_repair,
        decay,
        seg_length,
        op_coupling=op_coupling,
    )

    assert_equal(weights.num_destroy, num_destroy)
    assert_equal(weights.num_repair, num_repair)
    assert_equal(weights.op_coupling, op_coupling)


@mark.parametrize("decay", [1.01, -0.01, -0.5, 1.5])
def test_raises_invalid_decay(decay: float):
    with assert_raises(ValueError):
        SegmentedRouletteWheel([0, 0, 0, 0], 1, 1, decay)


@mark.parametrize("decay", np.linspace(0, 1, num=5))
def test_does_not_raise_valid_decay(decay: float):
    SegmentedRouletteWheel([0, 0, 0, 0], 1, 1, decay)


@mark.parametrize(
    "scores,decay,expected",
    [
        ([5, 3, 2, 1], 0, [3, 3]),  # 1 * 0 + (0 + 3) * 1 = 3
        ([5, 3, 2, 1], 1, [1, 1]),  # 1 * 1 + (0 + 3) * 0 = 1
        ([5, 3, 2, 1], 0.5, [2, 2]),  # .5 * 1 + (0 + 3) * .5 = 2
        ([5, 5, 5, 5], 0, [5, 5]),  # etc. etc.
        ([5, 5, 5, 5], 1, [1, 1]),
        ([5, 5, 5, 5], 0.5, [3, 3]),
    ],
)
def test_update(scores: List[float], decay: float, expected: List[float]):
    rnd_state = np.random.RandomState(1)
    weights = SegmentedRouletteWheel(scores, 1, 1, decay, 1)

    # TODO other weights?
    weights.update(Zero, 0, 0, 1)
    weights(rnd_state)

    assert_almost_equal(weights.destroy_weights[0], expected[0])
    assert_almost_equal(weights.repair_weights[0], expected[1])


@mark.parametrize(
    "scores,num_destroy,num_repair,decay,seg_length",
    [
        ([5, 3, 2, -1], 1, 1, 0, 1),  # negative score
        ([5, 3, 2], 1, 1, 0, 1),  # len(score) < 4
        ([5, 3, 2, 1], 0, 1, 0, 1),  # no destroy operator
        ([5, 3, 2, 1], 1, 0, 0, 1),  # no repair operator
        ([5, 3, 2, 1], 1, 1, -1, 1),  # decay < 0
        ([5, 3, 2, 1], 1, 1, 2, 1),  # decay > 1
        ([5, 3, 2, 1], 1, 1, 0.5, 0),  # seg_length < 1
    ],
)
def test_raises_invalid_arguments(
    scores: List[float],
    num_destroy: int,
    num_repair: int,
    decay: float,
    seg_length: int,
):
    with assert_raises(ValueError):
        SegmentedRouletteWheel(
            scores, num_destroy, num_repair, decay, seg_length
        )


# TODO test select weights, at iteration start
