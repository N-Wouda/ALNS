from typing import List

import numpy as np
from numpy.testing import assert_almost_equal, assert_raises
from pytest import mark

from alns.weights import SegmentedWeights


@mark.parametrize("seg_decay", [1.01, -0.01, -0.5, 1.5])
def test_raises_invalid_seg_decay(seg_decay: float):
    with assert_raises(ValueError):
        SegmentedWeights([0, 0, 0, 0], 1, 1, seg_decay)


@mark.parametrize("seg_decay", np.linspace(0, 1, num=5))
def test_does_not_raise_valid_seg_decay(seg_decay: float):
    SegmentedWeights([0, 0, 0, 0], 1, 1, seg_decay)


@mark.parametrize(
    "scores,seg_decay,expected",
    [
        ([5, 3, 2, 1], 0, [3, 3]),  # 1 * 0 + (0 + 3) * 1 = 3
        ([5, 3, 2, 1], 1, [1, 1]),  # 1 * 1 + (0 + 3) * 0 = 1
        ([5, 3, 2, 1], 0.5, [2, 2]),  # .5 * 1 + (0 + 3) * .5 = 2
        ([5, 5, 5, 5], 0, [5, 5]),  # etc. etc.
        ([5, 5, 5, 5], 1, [1, 1]),
        ([5, 5, 5, 5], 0.5, [3, 3]),
    ],
)
def test_update_weights(
    scores: List[float], seg_decay: float, expected: List[float]
):
    rnd_state = np.random.RandomState(1)
    weights = SegmentedWeights(scores, 1, 1, seg_decay, 1)
    op_coupling = np.ones((1, 1))

    # TODO other weights?
    weights.update_weights(0, 0, 1)
    weights.select_operators(rnd_state, op_coupling)

    assert_almost_equal(weights.destroy_weights[0], expected[0])
    assert_almost_equal(weights.repair_weights[0], expected[1])


@mark.parametrize(
    "scores,num_destroy,num_repair,seg_decay,seg_length",
    [
        ([5, 3, 2, -1], 1, 1, 0, 1),  # negative score
        ([5, 3, 2], 1, 1, 0, 1),  # len(score) < 4
        ([5, 3, 2, 1], 0, 1, 0, 1),  # no destroy operator
        ([5, 3, 2, 1], 1, 0, 0, 1),  # no repair operator
        ([5, 3, 2, 1], 1, 1, -1, 1),  # seg_decay < 0
        ([5, 3, 2, 1], 1, 1, 2, 1),  # seg_decay > 1
        ([5, 3, 2, 1], 1, 1, 0.5, 0),  # seg_length < 1
    ],
)
def test_raises_invalid_arguments(
    scores: List[float],
    num_destroy: int,
    num_repair: int,
    seg_decay: float,
    seg_length: int,
):
    with assert_raises(ValueError):
        SegmentedWeights(
            scores, num_destroy, num_repair, seg_decay, seg_length
        )


# TODO test select weights, at iteration start
