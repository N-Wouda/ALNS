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
    "scores, decay, num_destroy, num_repair, op_coupling",
    [
        ([0, 0, 0, 0], 0, 1, 1, np.ones((1, 1))),
        ([0, 1, 2, 3], 0.2, 2, 2, np.ones((2, 2))),
        ([5, 3, 2, 1], 1, 10, 10, np.ones((10, 10))),
    ],
)
def test_properties(scores, decay, num_destroy, num_repair, op_coupling):
    select = RouletteWheel(
        scores, decay, num_destroy, num_repair, op_coupling=op_coupling
    )

    # TODO move these property tests to RandomSelect
    assert_equal(select.num_destroy, num_destroy)
    assert_equal(select.num_repair, num_repair)
    assert_equal(select.op_coupling, op_coupling)

    assert_equal(select.scores, scores)
    assert_equal(select.destroy_weights, np.ones(num_destroy))
    assert_equal(select.repair_weights, np.ones(num_repair))
    assert_equal(select.decay, decay)


@mark.parametrize("decay", [1.01, -0.01, -0.5, 1.5])
def test_raises_invalid_decay(decay: float):
    with assert_raises(ValueError):
        RouletteWheel([0, 0, 0, 0], decay, 1, 1)


@mark.parametrize("decay", np.linspace(0, 1, num=5))
def test_does_not_raise_valid_decay(decay: float):
    RouletteWheel([0, 0, 0, 0], decay, 1, 1)


@mark.parametrize(
    "scores,decay,expected",
    [
        ([0, 0, 0, 0], 1, [1, 1]),  # scores are not used
        ([0, 0, 0, 0], 0, [0, 0]),  # initial weights are not used
        ([0.5, 0.5, 0.5, 0.5], 0.5, [0.75, 0.75]),
    ],
)  # convex combination
def test_update(scores: List[float], decay: float, expected: List[float]):
    select = RouletteWheel(scores, decay, 1, 1)

    # TODO other weights?
    select.update(Zero(), 0, 0, 1)

    assert_almost_equal(select.destroy_weights[0], expected[0])
    assert_almost_equal(select.repair_weights[0], expected[1])


@mark.parametrize(
    "op_coupling",
    [
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    ],
)
def test_select_coupled_operators(op_coupling):
    """
    Test if the indices of the selected operators correspond to the
    ones that are given by the operator coupling.
    """
    rng = rnd.default_rng()
    n_destroy, n_repair = op_coupling.shape
    select = RouletteWheel(
        [0, 0, 0, 0], 0, n_destroy, n_repair, op_coupling=op_coupling
    )
    d_idx, r_idx = select(rng, Zero(), Zero())

    assert_((d_idx, r_idx) in np.argwhere(op_coupling == 1))


@mark.parametrize(
    "op_coupling", [np.zeros((2, 2)), np.array([[0, 0, 0], [1, 1, 1]])]
)
def test_raise_uncoupled_destroy_op(op_coupling):
    """
    Tests if having a destroy operator that is not coupled to any of the
    repair operators raises an an error.
    """
    with assert_raises(ValueError):
        n_destroy, n_repair = op_coupling.shape
        RouletteWheel(
            [0, 0, 0, 0], 0, n_destroy, n_repair, op_coupling=op_coupling
        )


@mark.parametrize(
    "n_destroy, n_repair, op_coupling",
    [
        (1, 2, [0]),  # missing repair column
        (2, 2, [0, 0]),  # missing destroy row
        (2, 1, [0, 0]),  # too many repair, too few destroy
    ],
)
def test_raises_wrong_op_coupling_shape(n_destroy, n_repair, op_coupling):
    with assert_raises(ValueError):
        RouletteWheel([0, 0, 0, 0], 0, n_destroy, n_repair, op_coupling)


def test_single_destroy_operator_coerces_coupling_matrix():
    select = RouletteWheel([0, 0, 0, 0], 0, 1, 2, [1, 0])
    assert_equal(select.op_coupling.shape, (1, 2))
