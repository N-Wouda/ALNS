import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import LateAcceptanceHillClimbing
from alns.tests.states import Zero, One, Two


@mark.parametrize("n_iterations", [-0.01, -10, 1.5])
def test_raises_invalid_n_iterations(n_iterations):
    with assert_raises(ValueError):
        LateAcceptanceHillClimbing(n_iterations=n_iterations)


@mark.parametrize(
    "n_iterations, on_improve, only_better",
    [
        (1, True, True),
        (10, False, True),
        (100, True, False),
        (1000, False, False),
    ],
)
def test_properties(n_iterations, on_improve, only_better):
    lahc = LateAcceptanceHillClimbing(n_iterations, on_improve, only_better)

    assert_equal(lahc.n_iterations, n_iterations)
    assert_equal(lahc.only_better, only_better)
    assert_equal(lahc.on_improve, on_improve)


@mark.parametrize("n_iterations", [3, 10, 50])
def test_lahc_accept(n_iterations):
    """
    Tests if LAHC accepts a solution that is better than the current solution
    n_iterations iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(n_iterations, False, False)

    for _ in range(n_iterations):
        assert_(lahc(None, None, Two(), One()))

    # The current solution n_iterations iterations ago has value 2, so the
    # candidate solution with value 1 should be accepted despite being worse
    # than the current solution with value 0.
    assert_(lahc(None, None, Zero(), One()))


@mark.parametrize("n_iterations", [3, 10, 50])
def test_lahc_reject(n_iterations):
    """
    Tests if LAHC rejects a solution that is worse than the current solution
    n_iterations iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(n_iterations, False, False)

    for _ in range(n_iterations):
        assert_(lahc(None, None, One(), Zero()))

    # The previous current solution n_iterations ago has value 1, so the
    # candidate solution with value 1 should be rejected despite being better
    # than the current solution with value 2.
    assert_(not lahc(None, None, Two(), One()))


@mark.parametrize("n_iterations", [3, 10, 50])
def test_lahc_on_improve_accept(n_iterations):
    """
    Tests if LAHC criterion with on_improve=True accepts a solution that
    is better than the current solution despite being worse than the
    previous current solution from n_iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(n_iterations, True, False)

    for _ in range(n_iterations):
        assert_(not lahc(None, None, One(), Two()))

    # The previous current solution n_iterations ago has value 1 and the
    # candidate solution has value 1. But since the current solution has
    # value 2, the improved variant of LAHC will accept this solution.
    assert_(lahc(None, None, Two(), One()))


@mark.parametrize("n_iterations", [3, 10, 50])
def test_lahc_only_better_reject(n_iterations):
    """
    Tests if LAHC criterion with only_better=True rejects a solution that
    is better than the current solution despite being worse than the
    previous current solution from n_iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(n_iterations, False, True)

    for _ in range(n_iterations):
        assert_(not lahc(None, None, One(), Two()))

    for _ in range(n_iterations):
        # The current solutions are not stored because they are worse
        # than the previous current solutions
        assert_(not lahc(None, None, Two(), Two()))

    assert_(not lahc(None, None, Two(), One()))
