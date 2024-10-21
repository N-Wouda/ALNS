import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import LateAcceptanceHillClimbing
from alns.tests.states import One, Two, Zero


@mark.parametrize("lookback_period", [-0.01, -10, 1.5])
def test_raises_invalid_lookback_period(lookback_period):
    with assert_raises((ValueError, TypeError)):
        LateAcceptanceHillClimbing(lookback_period=lookback_period)


@mark.parametrize(
    "lookback_period, greedy, better_history",
    [
        (0, True, True),
        (1, False, True),
        (10, True, False),
        (100, False, False),
    ],
)
def test_properties(lookback_period, greedy, better_history):
    lahc = LateAcceptanceHillClimbing(lookback_period, greedy, better_history)

    assert_equal(lahc.lookback_period, lookback_period)
    assert_equal(lahc.better_history, better_history)
    assert_equal(lahc.greedy, greedy)


@mark.parametrize(
    "greedy, better_history",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_zero_lookback_period(greedy, better_history):
    """
    Test if LAHC behaves like regular hill climbing when `lookback_period` is
    set to zero.
    """
    lahc = LateAcceptanceHillClimbing(0, greedy, better_history)

    assert_(lahc(rnd.default_rng(), Zero(), Two(), One()))
    assert_(not lahc(rnd.default_rng(), Zero(), One(), One()))
    assert_(not lahc(rnd.default_rng(), Zero(), Zero(), Zero()))
    assert_(lahc(rnd.default_rng(), Zero(), Two(), One()))


@mark.parametrize("lookback_period", [0, 3, 10, 50])
def test_accept(lookback_period):
    """
    Tests if LAHC accepts a solution that is better than the current solution
    from `lookback_period` iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(lookback_period, False, False)

    assert_(lahc(rnd.default_rng(), Zero(), Two(), One()))

    for _ in range(lookback_period):
        # The then-current solution `lookback_period` iterations ago is 2, so
        # the candidate solution with value 1 should be accepted.
        assert_(lahc(rnd.default_rng(), Zero(), One(), One()))


@mark.parametrize("lookback_period", [0, 3, 10, 50])
def test_reject(lookback_period):
    """
    Tests if LAHC rejects a solution that is worse than the current solution
    from `lookback_period` iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(lookback_period, False, False)

    for _ in range(lookback_period):
        assert_(lahc(rnd.default_rng(), Zero(), One(), Zero()))

    # The then-current solution from `lookback_period` iterations ago has
    # value 1, so the candidate solution with value 1 should be rejected.
    assert_(not lahc(rnd.default_rng(), Zero(), Zero(), One()))


@mark.parametrize("lookback_period", [0, 3, 10, 50])
def test_greedy_accept(lookback_period):
    """
    Tests that if `greedy` is set, then a solution that is better than
    the current solution is accepted, despite being worse than the current
    solution from `lookback_period` iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(lookback_period, True, False)

    assert_(not lahc(rnd.default_rng(), Zero(), Zero(), Two()))

    for _ in range(lookback_period):
        # The candidate solution (1) is accepted because it is better than the
        # current solution (2), despite being worse than the compared
        # historial solution from `lookback_period` iterations ago (1).
        assert_(lahc(rnd.default_rng(), Zero(), Two(), One()))


def test_better_history_small_example():
    """
    Tests if only current solutions are stored that are better than
    the compared previous solution when `better_history` is set.
    """
    lahc = LateAcceptanceHillClimbing(1, False, True)

    assert_(lahc(rnd.default_rng(), Zero(), One(), Zero()))
    assert_(lahc(rnd.default_rng(), Zero(), Two(), Zero()))

    # Previous current stays at 1 because 2 was not better
    assert_(not lahc(rnd.default_rng(), Zero(), Zero(), One()))

    # Previous current is updated to Zero
    assert_(not lahc(rnd.default_rng(), Zero(), Zero(), Zero()))


@mark.parametrize("lookback_period", [3, 10, 50])
def test_better_history_reject(lookback_period):
    """
    Tests that if `better_history` is set, a solution can be rejected despite
    being better than the actual current solution from `lookback_period`
    iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(lookback_period, False, True)

    for _ in range(lookback_period):
        assert_(not lahc(rnd.default_rng(), Zero(), One(), Two()))

    for _ in range(lookback_period):
        # The current solutions are not stored because they are worse
        # than the historial current solutions
        assert_(not lahc(rnd.default_rng(), Zero(), Two(), Two()))

    for _ in range(lookback_period):
        # The candidates (1) do not improve the historical solutions (1)
        assert_(not lahc(rnd.default_rng(), Zero(), Two(), One()))


def test_full_example():
    """
    A full example to illustrate the LAHC criterion with a single
    lookback_period, greedy acceptance and better history management.
    """
    lahc = LateAcceptanceHillClimbing(1, True, True)

    # The first iteration compares candidate (1) against current (2)
    # and stores current in the history.
    assert_(lahc(rnd.default_rng(), Zero(), Two(), One()))

    # The second candidate (1) is accepted based on late-acceptance (2).
    # The historical value is updated to 1.
    assert_(lahc(rnd.default_rng(), Zero(), One(), One()))

    # The third candidate (1) is accepted based on greedy comparison with
    # the current solution (2). The historical value is not updated because
    # the current does not improve the historical value (1).
    assert_(lahc(rnd.default_rng(), Zero(), Two(), One()))

    # The fourth candidate (1) is not accepted because it does not improve
    # the current (1) nor does it improve the historical value (1).
    assert_(not lahc(rnd.default_rng(), Zero(), One(), One()))
