import numpy as np
import numpy.random as rnd
from numpy.testing import assert_, assert_allclose

from alns.select import RandomSelect
from alns.tests.states import Zero


def test_op_coupling():
    rnd_state = rnd.RandomState(1)

    # For i in {1..5}, each destroy operator i is coupled with repair operator
    # i. So only (i, i) pairs can be selected.
    op_coupling = np.eye(5)
    select = RandomSelect(5, 5, op_coupling)

    for _ in range(1_000):
        d_idx, r_idx = select(rnd_state, Zero(), Zero())
        assert_(d_idx == r_idx)


def test_uniform_selection():
    rnd_state = rnd.RandomState(1)
    histogram = np.zeros((2, 2))

    select = RandomSelect(2, 2)

    for _ in range(10_000):
        d_idx, r_idx = select(rnd_state, Zero(), Zero())
        histogram[d_idx, r_idx] += 1

    # There are four operator pair combinations, so each pair should have a
    # one in four chance of being selected. We allow a 0.005 margin since this
    # is based on sampling.
    histogram /= histogram.sum()
    assert_allclose(histogram, 0.25, atol=0.005)
