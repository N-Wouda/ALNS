import numpy as np
import numpy.random as rnd
from numpy.testing import (
    assert_,
    assert_equal,
    assert_raises,
)
from pytest import mark

from alns.accept import GreatDeluge
from alns.tests.states import One, Zero, Two


def test_raise_invalid_parameters():
    with assert_raises(ValueError):
        GreatDeluge(1, 0.5)  # alpha cannot be <= 1

    with assert_raises(ValueError):
        GreatDeluge(2, 1)  # beta cannot be outside of (0, 1)


def test_does_not_raise():
    GreatDeluge(2, 0.5)


@mark.parametrize("alpha", np.arange(1.1, 10, 1))
def test_alpha(alpha):
    great_deluge = GreatDeluge(alpha, 0.1)
    assert_equal(great_deluge.alpha, alpha)


@mark.parametrize("beta", np.arange(0.1, 1, 0.1))
def test_beta(beta):
    great_deluge = GreatDeluge(2, beta)
    assert_equal(great_deluge.beta, beta)


def test_threshold_update():
    """
    Test if the threshold is correctly updated using the linear update method.
    """
    great_deluge = GreatDeluge(2, 0.01)

    # The first candidate is rejected and increases the threshold
    great_deluge(None, Zero(), Zero(), One())
    assert_equal(great_deluge._threshold, 0.01)

    # The second candidate is accepted and decreases the threshold
    great_deluge(None, Zero(), Zero(), Zero())
    assert_equal(great_deluge._threshold, 0.0099)


def test_accepts_below_threshold():
    great_deluge = GreatDeluge(2.01, 0.5)

    # Initial threshold is set at 2.01, hence Two should be accepted
    assert_(great_deluge(None, One(), Zero(), Two()))


def test_rejects_above_threshold():
    great_deluge = GreatDeluge(1.01, 0.5)

    # Initial threshold is set at 1.01, hence Two should be rejected
    assert_(not great_deluge(None, One(), Zero(), Two()))


def test_rejects_equal_threshold():
    great_deluge = GreatDeluge(2, 0.5)

    # Initial threshold is set at 2, hence One should be rejected
    assert_(not great_deluge(None, One(), Zero(), Two()))
