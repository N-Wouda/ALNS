import pytest
from numpy.random import RandomState
from numpy.testing import assert_, assert_equal, assert_raises

from alns.tests.states import One, Zero, Two

from alns.stop import NoImprovement


@pytest.mark.parametrize("max_iterations", [-10, -100, -1000])
def test_raise_negative_parameters(max_iterations: int):
    """
    max_iterations cannot be negative.
    """
    with assert_raises(ValueError):
        NoImprovement(max_iterations)


@pytest.mark.parametrize("max_iterations", [0, 10, 100, 1000])
def test_does_not_raise(max_iterations: int):
    """
    Non-negative integers should not raise.
    """
    NoImprovement(max_iterations)


def test_max_iterations():
    """
    Test if the max_iterations parameter is correctly set.
    """
    stop = NoImprovement(3)
    assert_equal(stop.max_iterations, 3)


def test_zero_max_iterations():
    """
    Test if setting max_iterations to zero stops when a non-improving
    best solution has been found.
    """
    stop = NoImprovement(0)
    rnd = RandomState()

    assert_(not stop(rnd, One(), Zero()))
    assert_(not stop(rnd, Zero(), Zero()))
    assert_(stop(rnd, Zero(), Zero()))


def test_first_iteration():
    """
    Test if the first iteration does not stop.
    """
    stop = NoImprovement(100)
    rnd = RandomState()

    assert_(not stop(rnd, Zero(), One()))


def test_stop_after_max_iterations():
    stop = NoImprovement(100)
    rnd = RandomState()

    for _ in range(100):
        assert_(not stop(rnd, Zero(), Zero()))

    for _ in range(100):
        assert_(stop(rnd, Zero(), Zero()))


def test_counter_value_at_stop():
    """
    Test if the counter value is equal to the passed-in max iterations
    at the first stopping occurence.
    """
    stop = NoImprovement(100)

    while not stop(RandomState(), Zero(), Zero()):
        pass

    assert_equal(stop._counter, 100)
