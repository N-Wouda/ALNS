import numpy as np
import numpy.random as rnd
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)
from pytest import mark

from alns.accept import GreatDeluge
from alns.tests.states import One, Zero, Two, Three


def test_raise_invalid_parameters():
    with assert_raises(ValueError):
        GreatDeluge(1, 0.5)  # alpha cannot be <= 1

    with assert_raises(ValueError):
        GreatDeluge(2, 1)  # beta cannot be outside of (0, 1)

    with assert_raises(ValueError):
        # If method=non-linear, then gamma cannot be <= 0
        GreatDeluge(2, 0.5, 0, 0.5, method="non-linear")

    with assert_raises(ValueError):
        # If method=non-linear, then delta cannot be <= 0
        GreatDeluge(2, 0.5, 0.5, 0, method="non-linear")


def test_does_not_raise():
    GreatDeluge(2, 0.5)

    # Passing in gamma and delta without method='non-linear' should work
    GreatDeluge(2, 0.5, 0, 0)

    GreatDeluge(2, 0.5, 0.5, 0.5, method="non-linear")


# TODO Test properties if included?

"""
Tests for linear variant
"""


def test_linear_threshold_update():
    """
    Test if the threshold is correctly updated using the linear update method.
    """
    great_deluge = GreatDeluge(2, 0.01, method="linear")

    # The first candidate is rejected and increases the threshold
    great_deluge(None, Zero(), Zero(), One())
    assert_equal(great_deluge._threshold, 0.01)

    # The second candidate is accepted and decreases the threshold
    great_deluge(None, Zero(), Zero(), Zero())
    assert_equal(great_deluge._threshold, 0.0099)


def test_linear_accepts_below_threshold():
    great_deluge = GreatDeluge(2.01, 0.5, method="linear")

    # Initial threshold is set at 2.01, hence Two should be accepted
    assert_(great_deluge(None, One(), Zero(), Two()))


def test_linear_rejects_above_threshold():
    great_deluge = GreatDeluge(1.01, 0.5, method="linear")

    # Initial threshold is set at 1.01, hence Two should be rejected
    assert_(not great_deluge(None, One(), Zero(), Two()))


def test_linear_rejects_equal_threshold():
    great_deluge = GreatDeluge(2, 0.5, method="linear")

    # Initial threshold is set at 2, hence One should be rejected
    assert_(not great_deluge(None, One(), Zero(), Two()))


"""
Tests for non-linear variant
"""


def test_non_linear_raise_zero_best():
    """
    Test if an error is raised when the initial solution has value 0 and
    the non-linear variant is used.

    Reason: the relative gap cannot be computed.
    """
    great_deluge = GreatDeluge(2, 0.1, 0.01, 1, method="non-linear")

    with assert_raises(ValueError):
        great_deluge(None, Zero(), One(), One())


def test_non_linear_threshold_update():
    """
    Test if the threshold is correctly updated using the non-linear update
    method.
    """
    great_deluge = GreatDeluge(1.5, 0.1, 0.02, 1, method="non-linear")

    # The first candidate is rejected and linearly increases the threshold
    great_deluge(None, One(), Zero(), Two())
    assert_equal(great_deluge._threshold, 1.51)

    # The second candidate is accepted and exponentially decreases the threshold
    great_deluge(None, One(), Zero(), Zero())
    new_threshold = 1.51 * np.exp(-1) + 1
    assert_equal(great_deluge._threshold, new_threshold)


def test_non_linear_accepts_below_threshold():
    great_deluge = GreatDeluge(1.01, 0.5, 0.01, 1, method="non-linear")

    # Initial threshold is set at 1.01, hence One should be accepted
    assert_(great_deluge(None, One(), Zero(), One()))


def test_non_linear_rejects_above_threshold():
    great_deluge = GreatDeluge(2, 0.5, 1, 1, method="non-linear")

    # Initial threshold is set at 2 and the current solution is Zero,
    # hence Two should be rejected
    assert_(not great_deluge(None, One(), Zero(), Two()))


def test_non_linear_accepts_improving_current():
    great_deluge = GreatDeluge(2, 0.5, 1, 1, method="non-linear")

    # Two does not improve over the threshold (2) but does improve over the
    # current solution value (3), hence One should be accepted
    assert_(great_deluge(None, One(), Three(), Two()))
