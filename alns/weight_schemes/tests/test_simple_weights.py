from typing import List

import numpy as np
from numpy.testing import assert_raises, assert_almost_equal
from pytest import mark

from alns.weight_schemes import SimpleWeights


@mark.parametrize("op_decay", [1.01, -0.01, -0.5, 1.5])
def test_raises_invalid_op_decay(op_decay: float):
    with assert_raises(ValueError):
        SimpleWeights([0, 0, 0, 0], 1, 1, op_decay)


@mark.parametrize("op_decay", np.linspace(0, 1, num=5))
def test_does_not_raise_valid_op_decay(op_decay: float):
    SimpleWeights([0, 0, 0, 0], 1, 1, op_decay)


@mark.parametrize("scores,op_decay,expected",
                  [([0, 0, 0, 0], 1, [1, 1]),  # scores are not used
                   ([0, 0, 0, 0], 0, [0, 0]),  # initial weights are not used
                   ([.5, .5, .5, .5], .5, [.75, .75])])  # convex combination
def test_update_weights(scores: List[float],
                        op_decay: float,
                        expected: List[float]):
    weights = SimpleWeights(scores, 1, 1, op_decay)

    # TODO other weights?
    weights.update_weights(0, 0, 1)

    assert_almost_equal(weights.destroy_weights[0], expected[0])
    assert_almost_equal(weights.repair_weights[0], expected[1])


# TODO test select weights
