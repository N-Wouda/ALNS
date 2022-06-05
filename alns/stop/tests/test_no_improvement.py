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
    Test if setting max_iterations to zero always stops.
    """
    stop = NoImprovement(0)
    rnd = RandomState()

    assert_(stop(rnd, One(), Zero()))
    assert_(stop(rnd, Zero(), Zero()))


def test_one_max_iterations():
    """
    Test if setting max_iterations to one only stops when a non-improving
    best solution has been found.
    """
    stop = NoImprovement(1)
    rnd = RandomState()

    assert_(not stop(rnd, One(), Zero()))
    assert_(not stop(rnd, Zero(), Zero()))
    assert_(stop(rnd, Zero(), Zero()))


def test_n_max_iterations_non_improving():
    """
    Test if setting max_iterations to N correctly stops with non-improving
    solutions. The first N iterations should not stop. Beyond that, the
    the criterion should stop.
    """
    stop = NoImprovement(100)
    rnd = RandomState()

    for _ in range(100):
        assert_(not stop(rnd, Zero(), Zero()))

    for _ in range(100):
        assert_(stop(rnd, Zero(), Zero()))


def test_n_max_iterations_with_single_improvement():
    """
    Test if setting max_iterations to N correctly stops with a sequence
    of solutions, where the third solution is improving and the other solutions
    are non-improving. The first N+2 iterations should not stop. Beyond that,
    the criterion should stop.
    """
    stop = NoImprovement(100)
    rnd = RandomState()

    for _ in range(2):
        assert_(not stop(rnd, One(), Zero()))

    for _ in range(100):
        assert_(not stop(rnd, Zero(), Zero()))

    for _ in range(100):
        assert_(stop(rnd, Zero(), Zero()))
