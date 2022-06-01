import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import LateAcceptanceHillClimbing
from alns.tests.states import Zero, One, Two


@mark.parametrize("n_iterations", [-0.01, -10, 1.5])
def test_raises_invalid_n_iterations(n_iterations):
    with assert_raises(ValueError):
        LateAcceptanceHillClimbing(n_iterations)


@mark.parametrize("n_iterations", [3, 10, 50])
def test_late_acceptance(n_iterations):
    """
    Tests if the late acceptance hill climbing method accepts a solution
    that is better than the current solution n_iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(n_iterations)

    def accept(current, candidate):
        return lahc(None, None, current, candidate)

    accept(Two(), One())

    for _ in range(n_iterations - 1):
        accept(Zero(), One())

    # The current solution n_iterations ago has value 2, so the candidate
    # solution with value 1 should be accepted.
    assert_equal(lahc.history[0], 2)
    assert_(accept(Zero(), One()))


@mark.parametrize("n_iterations", [3, 10, 50])
def test_improved_late_acceptance(n_iterations):
    """
    Tests if the improved late acceptance hill climbing method accepts a
    solution that 1) is better than the current solution n_iterations ago
    or 2) is better than the current solution.
    """
    ilahc = LateAcceptanceHillClimbing(n_iterations, improved=True)

    def accept(current, candidate):
        return ilahc(None, None, current, candidate)

    accept(Zero(), One())

    for _ in range(n_iterations - 1):
        accept(Zero(), One())

    # The current solution n_iterations ago has value 0 and the candidate
    # solution has value 1. But since the current solution has value 1,
    # the improved variant of LAHC will accept this solution.
    assert_equal(ilahc.history[0], 0)
    assert_(accept(One(), One()))
