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

from alns.select import RouletteWheel
from alns.tests.states import Zero


@mark.parametrize(
    "scores, num_destroy, num_repair, decay, op_coupling",
    [
        ([0, 0, 0, 0], 1, 1, 0, np.ones((1, 1))),
        ([0, 1, 2, 3], 2, 2, 0.2, np.ones((2, 2))),
        ([5, 3, 2, 1], 10, 10, 1, np.ones((10, 10))),
    ],
)
def test_properties(scores, num_destroy, num_repair, decay, op_coupling):
    weights = RouletteWheel(
        scores, num_destroy, num_repair, decay, op_coupling=op_coupling
    )

    assert_equal(weights.num_destroy, num_destroy)
    assert_equal(weights.num_repair, num_repair)
    assert_equal(weights.op_coupling, op_coupling)


@mark.parametrize("op_decay", [1.01, -0.01, -0.5, 1.5])
def test_raises_invalid_op_decay(op_decay: float):
    with assert_raises(ValueError):
        RouletteWheel([0, 0, 0, 0], 1, 1, op_decay)


@mark.parametrize("op_decay", np.linspace(0, 1, num=5))
def test_does_not_raise_valid_op_decay(op_decay: float):
    RouletteWheel([0, 0, 0, 0], 1, 1, op_decay)


@mark.parametrize(
    "scores,op_decay,expected",
    [
        ([0, 0, 0, 0], 1, [1, 1]),  # scores are not used
        ([0, 0, 0, 0], 0, [0, 0]),  # initial weights are not used
        ([0.5, 0.5, 0.5, 0.5], 0.5, [0.75, 0.75]),
    ],
)  # convex combination
def test_update(scores: List[float], op_decay: float, expected: List[float]):
    select = RouletteWheel(scores, 1, 1, op_decay)

    # TODO other weights?
    select.update(Zero(), 0, 0, 1)

    assert_almost_equal(select.destroy_weights[0], expected[0])
    assert_almost_equal(select.repair_weights[0], expected[1])


# TODO Move the op_coupling related tests to RandomSelect
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
    select = RouletteWheel(
        [0, 0, 0, 0], n_destroy, n_repair, 0, op_coupling=op_coupling
    )
    d_idx, r_idx = select(rnd_state, Zero(), Zero())

    assert_((d_idx, r_idx) in np.argwhere(op_coupling == 1))


@mark.parametrize(
    "op_coupling", [np.zeros((3, 3)), np.array([[0, 0, 0], [1, 1, 1]])]
)
def test_raise_uncoupled_destroy_op(op_coupling):
    """
    Tests if having a destroy operator that is not coupled to any of the
    repair operators raises an an error.
    """
    with assert_raises(ValueError):
        n_destroy, n_repair = op_coupling.shape
        RouletteWheel(
            [0, 0, 0, 0], n_destroy, n_repair, 0, op_coupling=op_coupling
        )
