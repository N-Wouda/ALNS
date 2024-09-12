import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import MovingAverageThreshold
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
        MovingAverageThreshold(eta=eta, gamma=gamma)


@mark.parametrize("eta, gamma", [(1, 3), (0.4, 4)])
def test_no_raise_valid_parameters(eta, gamma):
    MovingAverageThreshold(eta=eta, gamma=gamma)


@mark.parametrize("eta", [0, 0.01, 0.5, 0.99, 1])
def test_eta(eta):
    moving_average = MovingAverageThreshold(eta, 3)
    assert_equal(moving_average.eta, eta)


@mark.parametrize("gamma", range(1, 10))
def test_gamma(gamma):
    moving_average = MovingAverageThreshold(0.5, gamma)
    assert_equal(moving_average.gamma, gamma)


def test_accepts_below_threshold():
    moving_average = MovingAverageThreshold(eta=0.5, gamma=4)
    moving_average(rnd.default_rng(), One(), One(), One())
    moving_average(rnd.default_rng(), One(), One(), Zero())

    # The threshold is set at 0 + 0.5 * (0.5 - 0) = 0.25
    assert_(moving_average(rnd.default_rng(), One(), One(), Zero()))


def test_rejects_above_threshold():
    moving_average = MovingAverageThreshold(eta=0.5, gamma=4)
    moving_average(rnd.default_rng(), One(), One(), Two())
    moving_average(rnd.default_rng(), One(), One(), Zero())

    # The threshold is set at 0 + 0.5 * (1 - 0) = 0.5
    assert_(not moving_average(rnd.default_rng(), One(), One(), One()))


def test_accepts_equal_threshold():
    moving_average = MovingAverageThreshold(eta=0.5, gamma=4)
    moving_average(rnd.default_rng(), One(), One(), VarObj(7100))
    moving_average(rnd.default_rng(), One(), One(), VarObj(7200))

    # The threshold is set at 7100 + 0.5 * (7140 - 7100) = 7120
    assert_(moving_average(rnd.default_rng(), One(), One(), VarObj(7120)))


def test_accepts_over_gamma_candidates():
    moving_average = MovingAverageThreshold(eta=0.2, gamma=3)
    moving_average(rnd.default_rng(), One(), One(), VarObj(7100))
    moving_average(rnd.default_rng(), One(), One(), VarObj(7200))
    moving_average(rnd.default_rng(), One(), One(), VarObj(7200))

    # The threshold is set at 7000 + 0.2 * (7133.33 - 7000) = 7013.33
    assert_(moving_average(rnd.default_rng(), One(), One(), VarObj(7000)))


def test_rejects_over_gamma_candidates():
    moving_average = MovingAverageThreshold(eta=0.2, gamma=3)

    for value in [7100, 7200, 7200, 7000]:
        moving_average(rnd.default_rng(), One(), One(), VarObj(value))

    # The threshold is set at 7000 + 0.2 * (7100 - 7000) = 7020
    result = moving_average(rnd.default_rng(), One(), One(), VarObj(7100))
    assert_(not result)


def test_evaluate_consecutive_solutions():
    """
    Test if MAT correctly accepts and rejects consecutive solutions.
    """
    moving_average = MovingAverageThreshold(eta=0.5, gamma=4)

    # The threshold is set at 7100, hence the solution is accepted.
    assert_(moving_average(rnd.default_rng(), One(), One(), VarObj(7100)))

    # The threshold is set at 7125, hence the solution is accepted.
    result = moving_average(rnd.default_rng(), One(), One(), VarObj(7200))
    assert_(not result)

    # The threshold is set at 7120, hence the solution is accepted.
    assert_(moving_average(rnd.default_rng(), One(), One(), VarObj(7120)))


def test_history():
    """
    Test if MAT correctly stores the history of the thresholds correctly.
    """
    moving_average = MovingAverageThreshold(eta=0.5, gamma=4)

    moving_average(rnd.default_rng(), One(), One(), VarObj(7100))
    assert_equal(moving_average.history, [7100])

    moving_average(rnd.default_rng(), One(), One(), VarObj(7200))
    assert_equal(moving_average.history, [7100, 7200])

    moving_average(rnd.default_rng(), One(), One(), VarObj(7120))
    assert_equal(moving_average.history, [7100, 7200, 7120])

    moving_average(rnd.default_rng(), One(), One(), VarObj(7100))
    assert_equal(moving_average.history, [7100, 7200, 7120, 7100])

    moving_average(rnd.default_rng(), One(), One(), VarObj(7200))
    assert_equal(moving_average.history, [7200, 7120, 7100, 7200])
