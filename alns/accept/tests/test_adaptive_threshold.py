import numpy as np
import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept.AdaptiveThreshold import AdaptiveThreshold
from alns.tests.states import One, Two, VarObj, Zero


@mark.parametrize(
    "eta, gamma",
    [
        (-1, 3),  # eta cannot be < 0 or > 1
        (0.5, -2),  # gamma cannot be < 0
    ],
)
def test_raise_invalid_parameters(eta, gamma):
    with assert_raises(ValueError):
        AdaptiveThreshold(eta=eta, gamma=gamma)


@mark.parametrize("eta, gamma", [(1, 3), (0.4, 4)])
def test_no_raise_valid_parameters(eta, gamma):
    AdaptiveThreshold(eta=eta, gamma=gamma)


@mark.parametrize("eta", np.arange(0, 1, 0.1))
def test_eta(eta):
    adaptive_threshold = AdaptiveThreshold(eta, 3)
    assert_equal(adaptive_threshold._eta, eta)


@mark.parametrize("gamma", np.arange(0, 10, 0.1))
def test_gamma(gamma):
    adaptive_threshold = AdaptiveThreshold(0.5, gamma)
    assert_equal(adaptive_threshold.gamma, gamma)


def test_accepts_below_threshold():
    adaptive_threshold = AdaptiveThreshold(eta=0.5, gamma=4)
    adaptive_threshold(rnd.RandomState(), One(), One(), One())
    adaptive_threshold(rnd.RandomState(), One(), One(), Zero())
    result = adaptive_threshold(rnd.RandomState(), One(), One(), Zero())

    # The threshold is set at 0 + 0.5 * (0.5 - 0) = 0.25
    assert_(result)


def test_rejects_above_threshold():
    adaptive_threshold = AdaptiveThreshold(eta=0.5, gamma=4)
    adaptive_threshold(rnd.RandomState(), One(), One(), Two())
    adaptive_threshold(rnd.RandomState(), One(), One(), Zero())
    result = adaptive_threshold(rnd.RandomState(), One(), One(), One())

    # The threshold is set at 0 + 0.5 * (1 - 0) = 0.5
    assert_(not result)


def test_accepts_equal_threshold():
    adaptive_threshold = AdaptiveThreshold(eta=0.5, gamma=4)
    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7100))
    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7200))
    result = adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7120))

    # The threshold is set at 7100 + 0.5 * (7140 - 7100) = 7120
    assert_(result)


def test_accepts_over_gamma_candidates():
    adaptive_threshold = AdaptiveThreshold(eta=0.2, gamma=3)
    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7100))
    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7200))
    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7200))
    result = adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7000))

    # The threshold is set at 7000 + 0.2 * (7133.33 - 7000) = 7013.33
    assert_(result)


def test_rejects_over_gamma_candidates():
    adaptive_threshold = AdaptiveThreshold(eta=0.2, gamma=3)
    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7100))
    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7200))
    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7200))
    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7000))
    result = adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7100))

    # The threshold is set at 7000 + 0.2 * (7100 - 7000) = 7020
    assert_(not result)


def test_evaluate_consecutive_solutions():
    """
    Test if AT correctly accepts and rejects consecutive solutions.
    """
    adaptive_threshold = AdaptiveThreshold(eta=0.5, gamma=4)

    result = adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7100))
    # The threshold is set at 7100, hence the solution is accepted
    assert_(result)

    result = adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7200))
    # The threshold is set at 7125, hence the solution is accepted
    assert_(not result)

    result = adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7120))
    # The threshold is set at 7120, hence the solution is accepted
    assert_(result)


def test_history():
    """
    Test if AT correctly stores the history of the thresholds correctly.
    """
    adaptive_threshold = AdaptiveThreshold(eta=0.5, gamma=4)

    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7100))
    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7200))
    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7120))
    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7100))
    adaptive_threshold(rnd.RandomState(), One(), One(), VarObj(7200))
    assert_(adaptive_threshold._history.__eq__([7200, 7120, 7100, 7200]))
