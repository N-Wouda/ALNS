import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import LateAcceptanceHillClimbing
from alns.tests.states import Zero, One, Two


@mark.parametrize("history_size", [-0.01, -10, 1.5])
def test_raises_invalid_history_size(history_size):
    with assert_raises(ValueError):
        LateAcceptanceHillClimbing(history_size)


@mark.parametrize("history_size", [3, 10, 50])
def test_late_acceptance(history_size):
    """
    Tests if the late acceptance hill climbing method with specified
    history size accepts a solution that is better than the current solution
    history_size iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(history_size)

    def accept(current, candidate):
        return lahc(rnd.RandomState(), Zero(), current, candidate)

    accept(Two(), One())

    for _ in range(history_size - 1):
        accept(Zero(), One())

    # 2 is the value to be compared against
    assert_equal(lahc.history[0], 2)
    assert_(accept(Zero(), One()))
    assert_equal(lahc.history[0], 0)


@mark.parametrize("history_size", [3, 10, 50])
def test_improved_late_acceptance(history_size):
    """
    Tests if the improved late acceptance hill climbing method with specified
    history size accepts a solution that 1) is better than the current solution
    history_size iterations ago or 2) is better than the current solution.
    """
    ilahc = LateAcceptanceHillClimbing(history_size, improved=True)

    def accept(current, candidate):
        return ilahc(rnd.RandomState(), Zero(), current, candidate)

    accept(Zero(), One())

    for _ in range(history_size - 1):
        accept(Zero(), One())

    # Compare against the historical current solution value (0)
    # or against the current solution (1)
    assert_equal(ilahc.history[0], 0)
    assert_(accept(One(), One()))
