import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import HillClimbing
from alns.tests.states import Zero, One, Two

"""
Hill Climbing
"""


def test_hc_accepts_better():
    """
    Tests if the hill climbing method accepts a better solution.
    """
    hill_climbing = HillClimbing()
    assert_(hill_climbing(rnd.RandomState(), One(), One(), Zero()))


def test_hc_rejects_worse():
    """
    Tests if the hill climbing method accepts a worse solution.
    """
    hill_climbing = HillClimbing()
    assert_(not hill_climbing(rnd.RandomState(), Zero(), Zero(), One()))


def test_hc_accepts_equal():
    """
    Tests if the hill climbing method accepts a solution that results in the
    same objective value.
    """
    hill_climbing = HillClimbing()
    assert_(hill_climbing(rnd.RandomState(), Zero(), Zero(), Zero()))


"""
Late Acceptance Hill Climbing
"""


@mark.parametrize("n_last", [-0.01, -10, 1.5])
def test_raises_invalid_n_last(n_last):
    with assert_raises(ValueError):
        HillClimbing(n_last=n_last)


@mark.parametrize("n_last", [3, 10, 50])
def test_late_acceptance(n_last):
    """
    Tests if the late acceptance hill climbing criterion accepts a solution
    that is better than the current solution n_last iterations ago.
    """
    lahc = HillClimbing(on_current=False, n_last=n_last)

    def accept(current, candidate):
        return lahc(None, None, current, candidate)

    accept(Two(), One())

    for _ in range(n_last - 1):
        accept(Zero(), One())

    # The current solution n_last iterations ago has value 2, so the candidate
    # solution with value 1 should be accepted.
    assert_equal(lahc._last_objectives[0], 2)
    assert_(accept(Zero(), One()))


@mark.parametrize("n_last", [3, 10, 50])
def test_improved_late_acceptance(n_last):
    """
    Tests if the improved late acceptance hill climbing criterion accepts a
    solution that 1) is better than the current solution n_last iterations ago
    or 2) is better than the current solution.
    """
    ilahc = HillClimbing(n_last=n_last)

    def accept(current, candidate):
        return ilahc(None, None, current, candidate)

    accept(Zero(), One())

    for _ in range(n_last - 1):
        accept(Zero(), One())

    # The current solution n_last iterations ago has value 0 and the candidate
    # solution has value 1. But since the current solution has value 1,
    # the improved variant of LAHC will accept this solution.
    assert_equal(ilahc._last_objectives[0], 0)
    assert_(accept(One(), One()))
