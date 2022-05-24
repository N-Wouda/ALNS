from typing import List

import numpy as np
import numpy.random as rnd
from numpy.testing import assert_, assert_raises, assert_almost_equal
from pytest import mark

from alns.weights import SimpleWeights


@mark.parametrize("op_decay", [1.01, -0.01, -0.5, 1.5])
def test_raises_invalid_op_decay(op_decay: float):
    with assert_raises(ValueError):
        SimpleWeights([0, 0, 0, 0], 1, 1, op_decay)


@mark.parametrize("op_decay", np.linspace(0, 1, num=5))
def test_does_not_raise_valid_op_decay(op_decay: float):
    SimpleWeights([0, 0, 0, 0], 1, 1, op_decay)


@mark.parametrize(
    "scores,op_decay,expected",
    [
        ([0, 0, 0, 0], 1, [1, 1]),  # scores are not used
        ([0, 0, 0, 0], 0, [0, 0]),  # initial weights are not used
        ([0.5, 0.5, 0.5, 0.5], 0.5, [0.75, 0.75]),
    ],
)  # convex combination
def test_update_weights(
    scores: List[float], op_decay: float, expected: List[float]
):
    weights = SimpleWeights(scores, 1, 1, op_decay)

    # TODO other weights?
    weights.update_weights(0, 0, 1)

    assert_almost_equal(weights.destroy_weights[0], expected[0])
    assert_almost_equal(weights.repair_weights[0], expected[1])


@mark.parametrize(
    "op_coupling",
    [
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),  # Not allowed by ALNS
    ],
)
def test_select_operators(op_coupling):
    """
    Test if the indices of the selected operators correspond to the
    ones that are given by the operator coupling.
    """
    rnd_state = rnd.RandomState()
    n_destroy, n_repair = op_coupling.shape
    weights = SimpleWeights([0, 0, 0, 0], n_destroy, n_repair, 0)
    d_idx, r_idx = weights.select_operators(rnd_state, op_coupling)

    assert_((d_idx, r_idx) in np.argwhere(op_coupling == 1))
