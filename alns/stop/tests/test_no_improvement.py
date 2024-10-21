import pytest
from numpy.random import default_rng
from numpy.testing import assert_, assert_equal, assert_raises

from alns.stop import NoImprovement
from alns.tests.states import One, Zero


@pytest.mark.parametrize("max_iterations", [-10, -100, -1000])
def test_raise_negative_parameters(max_iterations: int):
    """
    max_iterations cannot be negative.
    """
    with assert_raises(ValueError):
        NoImprovement(max_iterations)


@pytest.mark.parametrize("max_iterations", [0, 10, 100, 1000])
def test_max_iterations(max_iterations):
    """
    Test if the max_iterations parameter is correctly set.
    """
    stop = NoImprovement(max_iterations)
    assert_equal(stop.max_iterations, max_iterations)


def test_zero_max_iterations():
    """
    Test if setting max_iterations to zero always stops.
    """
    stop = NoImprovement(0)
    rng = default_rng()

    assert_(stop(rng, One(), Zero()))
    assert_(stop(rng, Zero(), Zero()))


def test_one_max_iterations():
    """
    Test if setting max_iterations to one only stops when a non-improving
    best solution has been found.
    """
    stop = NoImprovement(1)
    rng = default_rng()

    assert_(not stop(rng, One(), Zero()))
    assert_(not stop(rng, Zero(), Zero()))
    assert_(stop(rng, Zero(), Zero()))


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_n_max_iterations_non_improving(n):
    """
    Test if setting max_iterations to n correctly stops with non-improving
    solutions. The first n iterations should not stop. Beyond that, the
    the criterion should stop.
    """
    stop = NoImprovement(n)
    rng = default_rng()

    for _ in range(n):
        assert_(not stop(rng, Zero(), Zero()))

    for _ in range(n):
        assert_(stop(rng, Zero(), Zero()))


@pytest.mark.parametrize("n, k", [(10, 2), (100, 20), (1000, 200)])
def test_n_max_iterations_with_single_improvement(n, k):
    """
    Test if setting max_iterations to n correctly stops with a sequence
    of solutions, where the k-th solution is improving and the other solutions
    are non-improving. The first n+k-1 iterations should not stop. Beyond that,
    the criterion should stop.
    """
    stop = NoImprovement(n)
    rng = default_rng()

    for _ in range(k):
        assert_(not stop(rng, One(), Zero()))

    for _ in range(n):
        assert_(not stop(rng, Zero(), Zero()))

    for _ in range(n):
        assert_(stop(rng, Zero(), Zero()))
