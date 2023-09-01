import numpy as np
import numpy.random as rnd
from numpy.testing import assert_, assert_allclose, assert_approx_equal

from alns.select import RandomSelect
from alns.tests.states import Zero


def test_op_coupling():
    rng = rnd.default_rng(1)

    # For i in {1..5}, each destroy operator i is coupled with repair operator
    # i. So only (i, i) pairs can be selected.
    op_coupling = np.eye(5)
    select = RandomSelect(5, 5, op_coupling)

    for _ in range(1_000):
        d_idx, r_idx = select(rng, Zero(), Zero())
        assert_(d_idx == r_idx)


def test_uniform_selection():
    rng = rnd.default_rng(1)
    histogram = np.zeros((2, 2))

    select = RandomSelect(2, 2)

    for _ in range(10_000):
        d_idx, r_idx = select(rng, Zero(), Zero())
        histogram[d_idx, r_idx] += 1

    # There are four operator pair combinations, so each pair should have a
    # one in four chance of being selected. We allow a 0.01 margin since this
    # is based on sampling.
    histogram /= histogram.sum()
    assert_allclose(histogram, 0.25, atol=0.01)


def test_uniform_selection_op_coupling():
    rng = rnd.default_rng(1)
    histogram = np.zeros((2, 2))

    op_coupling = np.eye(2)
    op_coupling[0, 1] = 1

    select = RandomSelect(2, 2, op_coupling)

    for _ in range(10_000):
        d_idx, r_idx = select(rng, Zero(), Zero())
        histogram[d_idx, r_idx] += 1

    # There are three OK operator pair combinations, so each such pair should
    # have a one in three chance of being selected.
    histogram /= histogram.sum()

    # These should be sampled uniformly...
    assert_approx_equal(histogram[0, 0], 1 / 3, significant=2)
    assert_approx_equal(histogram[0, 1], 1 / 3, significant=2)
    assert_approx_equal(histogram[1, 1], 1 / 3, significant=2)

    # ...but this one's not allowed by the operator coupling matrix.
    assert_approx_equal(histogram[1, 0], 0, significant=7)


def test_single_operators():
    rng = rnd.default_rng(1)
    select = RandomSelect(1, 1)

    # Only one (destroy, repair) operator pair, so should return (0, 0).
    assert_(select(rng, Zero(), Zero()) == (0, 0))
