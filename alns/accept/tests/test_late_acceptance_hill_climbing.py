import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import LateAcceptanceHillClimbing
from alns.tests.states import Zero, One, Two


@mark.parametrize("history_length", [-0.01, -10, 1.5])
def test_raises_invalid_history_length(history_length):
    with assert_raises(ValueError):
        LateAcceptanceHillClimbing(history_length=history_length)


@mark.parametrize(
    "history_length, greedy, collect_better",
    [
        (1, True, True),
        (10, False, True),
        (100, True, False),
        (1000, False, False),
    ],
)
def test_properties(history_length, greedy, collect_better):
    lahc = LateAcceptanceHillClimbing(history_length, greedy, collect_better)

    assert_equal(lahc.history_length, history_length)
    assert_equal(lahc.collect_better, collect_better)
    assert_equal(lahc.greedy, greedy)


@mark.parametrize("history_length", [3, 10, 50])
def test_accept(history_length):
    """
    Tests if LAHC accepts a solution that is better than the current solution
    from history_length iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(history_length, False, False)

    for _ in range(history_length):
        assert_(lahc(rnd.RandomState(), Zero(), Two(), One()))

    # The previous current solution history_length ago has value 2, so the
    # candidate solution with value 1 should be accepted.
    assert_(lahc(rnd.RandomState(), Zero(), Zero(), One()))


@mark.parametrize("history_length", [3, 10, 50])
def test_reject(history_length):
    """
    Tests if LAHC rejects a solution that is worse than the current solution
    history_length iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(history_length, False, False)

    for _ in range(history_length):
        assert_(lahc(rnd.RandomState(), Zero(), One(), Zero()))

    # The compared previous current solution has value 1, so the
    # candidate solution with value 1 should be rejected.
    assert_(not lahc(rnd.RandomState(), Zero(), Two(), One()))


@mark.parametrize("history_length", [3, 10, 50])
def test_greedy_accept(history_length):
    """
    Tests if LAHC criterion with greedy=True accepts a solution that
    is better than the current solution despite being worse than the
    previous current solution from history_length iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(history_length, True, False)

    for _ in range(history_length):
        assert_(not lahc(rnd.RandomState(), Zero(), One(), Two()))

    # The candidate (1) is better than the current (2), hence it is accepted
    # despite being worse than the previous current (1).
    assert_(lahc(rnd.RandomState(), Zero(), Two(), One()))


def test_collect_better():
    """
    Tests if only current solutions are stored that are better than
    the compared previous solution when only_better=True.
    """
    lahc = LateAcceptanceHillClimbing(1, False, True)

    assert_(lahc(rnd.RandomState(), Zero(), One(), Zero()))

    # Previous current stays at 1 because 2 it not better
    assert_(not lahc(rnd.RandomState(), Zero(), Two(), One()))
    assert_(lahc(rnd.RandomState(), Zero(), Two(), Zero()))

    # Previous current updates to 0
    assert_(not lahc(rnd.RandomState(), Zero(), Zero(), Zero()))


@mark.parametrize("history_length", [3, 10, 50])
def test_collect_better_reject(history_length):
    """
    Tests if LAHC criterion with collect_better=True rejects a solution that
    is better than the previous current solution from history_length iterations
    ago, because that previous current solution was not better than the current
    solution from (2 * history_length) iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(history_length, False, True)

    for _ in range(history_length):
        assert_(not lahc(rnd.RandomState(), Zero(), One(), Two()))

    for _ in range(history_length):
        # The current solutions are not stored because they are worse
        # than the previous current solutions
        assert_(not lahc(rnd.RandomState(), Zero(), Two(), Two()))

    assert_(not lahc(rnd.RandomState(), Zero(), Two(), One()))
