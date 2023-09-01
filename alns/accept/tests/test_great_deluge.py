import numpy as np
import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import GreatDeluge
from alns.tests.states import One, Two, Zero


@mark.parametrize(
    "alpha, beta",
    [
        (1, 0.5),  # alpha cannot be <= 1
        (2, 1),  # beta cannot be outside of (0, 1)
    ],
)
def test_raise_invalid_parameters(alpha, beta):
    with assert_raises(ValueError):
        GreatDeluge(alpha, beta)


@mark.parametrize("alpha, beta", [(1.01, 0.01), (999, 0.5)])
def test_no_raise_valid_parameters(alpha, beta):
    GreatDeluge(alpha, beta)


@mark.parametrize("alpha", np.arange(1.1, 10, 1))
def test_alpha(alpha):
    great_deluge = GreatDeluge(alpha, 0.1)
    assert_equal(great_deluge.alpha, alpha)


@mark.parametrize("beta", np.arange(0.1, 1, 0.1))
def test_beta(beta):
    great_deluge = GreatDeluge(2, beta)
    assert_equal(great_deluge.beta, beta)


def test_accepts_below_threshold():
    great_deluge = GreatDeluge(2.01, 0.5)

    # Initial threshold is set at 2.01, hence Two should be accepted
    assert_(great_deluge(rnd.default_rng(), One(), Zero(), Two()))


def test_rejects_above_threshold():
    great_deluge = GreatDeluge(1.01, 0.5)

    # Initial threshold is set at 1.01, hence Two should be rejected
    assert_(not great_deluge(rnd.default_rng(), One(), Zero(), Two()))


def test_rejects_equal_threshold():
    great_deluge = GreatDeluge(2, 0.5)

    # Initial threshold is set at 2, hence Two should be rejected
    assert_(not great_deluge(rnd.default_rng(), One(), Zero(), Two()))


def test_evaluate_consecutive_solutions():
    """
    Test if GD correctly accepts and rejects consecutive solutions.
    """
    great_deluge = GreatDeluge(2, 0.01)

    # The initial threshold is set at 2*0 = 0, so the first candidate with
    # value one is rejected. The threshold is updated to 0.01.
    assert_(not great_deluge(rnd.default_rng(), Zero(), Zero(), One()))

    # The second candidate is below the threshold (0 < 0.01), hence accepted.
    # The threshold is updated to 0.0099.
    assert_(great_deluge(rnd.default_rng(), Zero(), Zero(), Zero()))

    # The third candidate is below the threshold (0 < 0.0099), hence accepted.
    assert_(great_deluge(rnd.default_rng(), Zero(), Zero(), Zero()))
