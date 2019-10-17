from numpy.random import RandomState
from numpy.testing import assert_equal

from alns.select_operator import select_operator


def test_single_choice():
    """
    When there is only a single choice, the method should return the same
    operator index every time, regardless of the weights and random state.
    """
    for _ in range(100):
        assert_equal(select_operator([None], [1], RandomState()), 0)


def test_all_weight_on_one_operator():
    """
    When all weight is assigned to a single operator, that operator must be
    selected every time.
    """
    for _ in range(100):        # all weigt is now on the first operator.
        assert_equal(select_operator([None, None], [5, 0], RandomState()), 0)

    for _ in range(100):        # and now on the second.
        assert_equal(select_operator([None, None], [0, 5], RandomState()), 1)
