from unittest.mock import Mock

import numpy.random as rnd
from numpy.testing import assert_equal, assert_raises
from pytest import mark

from alns.select import RandomSelect


@mark.parametrize(
    "num_destroy,num_repair,rnd_vals",
    [(1, 1, [0, 0]), (10, 1, [9, 0]), (1, 10, [0, 1]), (10, 10, [5, 5])],
)
def test_select_operator(num_destroy, num_repair, rnd_vals):
    (test_d_idx, test_r_idx) = rnd_vals

    rng = Mock(spec_set=rnd.RandomState, randint=lambda _: rnd_vals.pop(0))
    random_select = RandomSelect(num_destroy, num_repair)
    d_idx, r_idx = random_select.select_operators(rng)

    assert_equal(d_idx, test_d_idx)
    assert_equal(r_idx, test_r_idx)


@mark.parametrize(
    "num_destroy,num_repair",
    [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (-1, -1), (1, -1), (-1, 1)],
)
def test_raises_invalid_params(num_destroy, num_repair):
    with assert_raises(ValueError):
        RandomSelect(num_destroy, num_repair)
