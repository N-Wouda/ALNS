from typing import List

import numpy as np
from numpy.testing import assert_almost_equal, assert_raises
from pytest import mark

from alns.weight_schemes import SegmentedWeights


@mark.parametrize("seg_decay", [1.01, -0.01, -0.5, 1.5])
def test_raises_invalid_seg_decay(seg_decay: float):
    with assert_raises(ValueError):
        SegmentedWeights([0, 0, 0, 0], 1, 1, seg_decay)


@mark.parametrize("seg_decay", np.linspace(0, 1, num=5))
def test_does_not_raise_valid_seg_decay(seg_decay: float):
    SegmentedWeights([0, 0, 0, 0], 1, 1, seg_decay)


@mark.parametrize("scores,seg_decay,expected",
                  [([5, 3, 2, 1], 0, [3, 3]),  # 1 * 0 + (0 + 3) * 1 = 3
                   ([5, 3, 2, 1], 1, [1, 1]),  # 1 * 1 + (0 + 3) * 0 = 1
                   ([5, 3, 2, 1], .5, [2, 2]),  # .5 * 1 + (0 + 3) * .5 = 2
                   ([5, 5, 5, 5], 0, [5, 5]),  # etc. etc.
                   ([5, 5, 5, 5], 1, [1, 1]),
                   ([5, 5, 5, 5], .5, [3, 3])])
def test_update_weights(scores: List[float],
                        seg_decay: float,
                        expected: List[float]):
    weights = SegmentedWeights(scores, 1, 1, seg_decay, 1)

    # TODO other weights?
    weights.update_weights(0, 0, 1)
    weights.at_iteration_start(1, 1)

    assert_almost_equal(weights.destroy_weights[0], expected[0])
    assert_almost_equal(weights.repair_weights[0], expected[1])

# TODO test select weights, at iteration start