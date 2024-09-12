import pytest
from numpy.random import default_rng
from numpy.testing import assert_, assert_equal, assert_raises

from alns.stop import MaxIterations
from alns.tests.states import Zero


@pytest.mark.parametrize("max_iterations", [-1, -42, -10000])
def test_raise_negative_parameters(max_iterations: int):
    """
    Maximum iterations cannot be negative.
    """
    with assert_raises(ValueError):
        MaxIterations(max_iterations)


@pytest.mark.parametrize("max_iterations", [1, 42, 10000])
def test_does_not_raise(max_iterations: int):
    """
    Valid parameters should not raise.
    """
    MaxIterations(max_iterations)


@pytest.mark.parametrize("max_iterations", [1, 42, 10000])
def test_max_iterations(max_iterations):
    """
    Test if the max iterations parameter is correctly set.
    """
    stop = MaxIterations(max_iterations)
    assert_equal(stop.max_iterations, max_iterations)


def test_before_max_iterations():
    stop = MaxIterations(100)
    rng = default_rng(0)

    for _ in range(100):
        assert_(not stop(rng, Zero(), Zero()))


def test_after_max_iterations():
    stop = MaxIterations(100)
    rng = default_rng()

    for _ in range(100):
        stop(rng, Zero(), Zero())

    for _ in range(100):
        assert_(stop(rng, Zero(), Zero()))
