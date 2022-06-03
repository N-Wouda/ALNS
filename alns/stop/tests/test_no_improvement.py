import pytest
from numpy.random import RandomState
from numpy.testing import assert_, assert_equal, assert_raises

from alns.tests.states import One, Zero, Two

from alns.stop import NoImprovement


@pytest.mark.parametrize("n_iterations", [-10, -100, -1000])
def test_raise_negative_parameters(n_iterations: int):
    """
    n_iterations cannot be negative.
    """
    with assert_raises(ValueError):
        NoImprovement(n_iterations)


@pytest.mark.parametrize("n_iterations", [0, 10, 100, 1000])
def test_does_not_raise(n_iterations: int):
    """
    Non-negative integers should not raise.
    """
    NoImprovement(n_iterations)


def test_n_iterations():
    """
    Test if the n_iterations parameter is correctly set.
    """
    stop = NoImprovement(3)
    assert stop.n_iterations == 3


def test_first_iteration():
    """
    Test if the first iteration does not stop.
    """
    stop = NoImprovement(100)
    rnd = RandomState()

    assert not stop(rnd, Zero(), Zero())


def test_before_n_iterations():
    stop = NoImprovement(100)
    rnd = RandomState()

    for _ in range(100):
        assert_(not stop(rnd, Zero(), Zero()))


def test_after_n_iterations():
    stop = NoImprovement(100)
    rnd = RandomState()

    for _ in range(100 + 1):
        stop(rnd, Zero(), Zero())

    for _ in range(100):
        # The 102-th iteration and beyond should be stopped, because there was
        # 1 iteration to initialize and 100 subsequent non-improving iterations.
        assert_(stop(rnd, Zero(), Zero()))


@pytest.mark.parametrize("n_iterations", [10, 100, 1000])
def test_reset_counter(n_iterations):
    """
    Test if the counter is reset when an improving solution is encountered.
    """
    stop = NoImprovement(n_iterations)
    rnd = RandomState()

    for _ in range(n_iterations):
        assert_(not stop(rnd, Two(), Zero()))

    assert not stop(rnd, One(), Zero())
    assert stop._counter == 0
