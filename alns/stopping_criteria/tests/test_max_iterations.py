import pytest

from numpy.testing import assert_, assert_raises

from alns.stopping_criteria import MaxIterations
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
    assert stop.max_iterations == max_iterations


@pytest.mark.parametrize(
    "max_iterations, iterations", [(1, 0), (1000, 500), (0, 100)]
)
def test_current_iteration(max_iterations: int, iterations: int):
    """
    Test if the current iteration parameter is correctly set.
    """
    stop = MaxIterations(max_iterations)

    assert_(stop.current_iteration == 0)

    for _ in range(iterations):
        stop(Zero(), Zero())

    assert_(stop.current_iteration == iterations)


def test_before_max_iterations():
    stop = MaxIterations(100)

    for _ in range(100):
        assert_(not stop(Zero(), Zero()))


def test_after_max_iterations():
    stop = MaxIterations(100)

    for _ in range(100):
        stop(Zero(), Zero())

    for _ in range(100):
        assert_(stop(Zero(), Zero()))
