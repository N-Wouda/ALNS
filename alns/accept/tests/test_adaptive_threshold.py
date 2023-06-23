import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import AdaptiveThreshold
from alns.tests.states import One, Two, VarObj, Zero


@mark.parametrize(
    "eta, gamma",
    [
        (-1, 3),  # eta cannot be < 0
        (2, 3),  # eta cannot be > 1
        (0.5, -2),  # gamma cannot be < 0
        (0.5, 0),  # gamma cannot be 0
    ],
)
def test_raise_invalid_parameters(eta, gamma):
    with assert_raises(ValueError):
        AdaptiveThreshold(eta=eta, gamma=gamma)


@mark.parametrize("eta, gamma", [(1, 3), (0.4, 4)])
def test_no_raise_valid_parameters(eta, gamma):
    AdaptiveThreshold(eta=eta, gamma=gamma)


@mark.parametrize("eta", [0, 0.01, 0.5, 0.99, 1])
def test_eta(eta):
    adaptive_threshold = AdaptiveThreshold(eta, 3)
    assert_equal(adaptive_threshold.eta, eta)


@mark.parametrize("gamma", range(1, 10))
def test_gamma(gamma):
    adaptive_threshold = AdaptiveThreshold(0.5, gamma)
    assert_equal(adaptive_threshold.gamma, gamma)


def test_accepts_below_threshold():
    adaptive_threshold = AdaptiveThreshold(eta=0.5, gamma=4)
    adaptive_threshold(rnd.RandomState(), One(), One(), One())
    adaptive_threshold(rnd.RandomState(), One(), One(), Zero())

    # The threshold is set at 0 + 0.5 * (0.5 - 0) = 0.25
    assert_(adaptive_threshold(rnd.RandomState(), One(), One(), Zero()))


def test_rejects_above_threshold():
    adaptive_threshold = AdaptiveThreshold(eta=0.5, gamma=4)
    adaptive_threshold(rnd.RandomState(), One(), One(), Two())
    adaptive_threshold(rnd.RandomState(), One(), One(), Zero())

    # The threshold is set at 0 + 0.5 * (1 - 0) = 0.5
    assert_(not adaptive_threshold(rnd.RandomState(), One(), One(), One()))


def test_accepts_equal_threshold():
    accept = AdaptiveThreshold(eta=0.5, gamma=4)
    accept(rnd.RandomState(), One(), One(), VarObj(7100))
    accept(rnd.RandomState(), One(), One(), VarObj(7200))

    # The threshold is set at 7100 + 0.5 * (7140 - 7100) = 7120
    assert_(accept(rnd.RandomState(), One(), One(), VarObj(7120)))


def test_accepts_over_gamma_candidates():
    accept = AdaptiveThreshold(eta=0.2, gamma=3)
    accept(rnd.RandomState(), One(), One(), VarObj(7100))
    accept(rnd.RandomState(), One(), One(), VarObj(7200))
    accept(rnd.RandomState(), One(), One(), VarObj(7200))

    # The threshold is set at 7000 + 0.2 * (7133.33 - 7000) = 7013.33
    assert_(accept(rnd.RandomState(), One(), One(), VarObj(7000)))


def test_rejects_over_gamma_candidates():
    accept = AdaptiveThreshold(eta=0.2, gamma=3)

    for value in [7100, 7200, 7200, 7000]:
        accept(rnd.RandomState(), One(), One(), VarObj(value))

    # The threshold is set at 7000 + 0.2 * (7100 - 7000) = 7020
    result = accept(rnd.RandomState(), One(), One(), VarObj(7100))
    assert_(not result)


def test_evaluate_consecutive_solutions():
    """
    Test if AT correctly accepts and rejects consecutive solutions.
    """
    accept = AdaptiveThreshold(eta=0.5, gamma=4)

    # The threshold is set at 7100, hence the solution is accepted.
    assert_(accept(rnd.RandomState(), One(), One(), VarObj(7100)))

    # The threshold is set at 7125, hence the solution is accepted.
    result = accept(rnd.RandomState(), One(), One(), VarObj(7200))
    assert_(not result)

    # The threshold is set at 7120, hence the solution is accepted.
    assert_(accept(rnd.RandomState(), One(), One(), VarObj(7120)))


def test_history():
    """
    Test if AT correctly stores the history of the thresholds correctly.
    """
    accept = AdaptiveThreshold(eta=0.5, gamma=4)

    accept(rnd.RandomState(), One(), One(), VarObj(7100))
    assert_equal(accept.history, [7100])

    accept(rnd.RandomState(), One(), One(), VarObj(7200))
    assert_equal(accept.history, [7100, 7200])

    accept(rnd.RandomState(), One(), One(), VarObj(7120))
    assert_equal(accept.history, [7100, 7200, 7120])

    accept(rnd.RandomState(), One(), One(), VarObj(7100))
    assert_equal(accept.history, [7100, 7200, 7120, 7100])

    accept(rnd.RandomState(), One(), One(), VarObj(7200))
    assert_equal(accept.history, [7200, 7120, 7100, 7200])
