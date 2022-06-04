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

    with assert_raises(ValueError):
        GreatDeluge(2, 0.5, 0.5, 0.5, method="exponential")  # Invalid method


def test_does_not_raise():
    GreatDeluge(2, 0.5)

    # Passing in gamma and delta without method='non-linear' should work
    GreatDeluge(2, 0.5, 0, 0)

    GreatDeluge(2, 0.5, 0.5, 0.5, method="non-linear")


# TODO Test properties if included?

"""
Linear Great Deluge (GD)
"""


def test_gd_threshold_update():
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


def test_gd_accepts_below_threshold():
    great_deluge = GreatDeluge(2.01, 0.5, method="linear")

    # Initial threshold is set at 2.01, hence Two should be accepted
    assert_(great_deluge(None, One(), Zero(), Two()))


def test_gd_rejects_above_threshold():
    great_deluge = GreatDeluge(1.01, 0.5, method="linear")

    # Initial threshold is set at 1.01, hence Two should be rejected
    assert_(not great_deluge(None, One(), Zero(), Two()))


def test_gd_rejects_equal_threshold():
    great_deluge = GreatDeluge(2, 0.5, method="linear")

    # Initial threshold is set at 2, hence One should be rejected
    assert_(not great_deluge(None, One(), Zero(), Two()))


"""
Non-Linear Great Deluge
"""


def test_nlgd_raise_zero_best():
    """
    Test if an error is raised when the initial solution has value 0
    using NLGD.

    Reason: the relative gap cannot be computed with zero threshold.
    """
    great_deluge = GreatDeluge(2, 0.1, 0.01, 1, method="non-linear")

    with assert_raises(ValueError):
        great_deluge(None, Zero(), One(), One())


def test_nlgd_accepts_below_threshold():
    great_deluge = GreatDeluge(1.01, 0.5, 0.01, 1, method="non-linear")

    # Initial threshold is set at 1.01, hence One should be accepted
    assert_(great_deluge(None, One(), Zero(), One()))


def test_nlgd_rejects_above_threshold():
    great_deluge = GreatDeluge(2, 0.5, 1, 1, method="non-linear")

    # Initial threshold is set at 2 and the current solution is Zero,
    # hence Two should be rejected
    assert_(not great_deluge(None, One(), Zero(), Two()))


def test_nlgd_accepts_improving_current():
    great_deluge = GreatDeluge(2, 0.5, 1, 1, method="non-linear")

    # Candidate is does not improve the threshold (2 == 2) but does improve the
    # current solution value (2 < 3), hence candidate should be accepted
    assert_(great_deluge(None, One(), Three(), Two()))


def test_nlgd_evaluate_consecutive_solutions():
    """
    Test if NLGD correctly accepts and rejects consecutive solutions.
    """
    great_deluge = GreatDeluge(1.5, 0.1, 0.02, 2, method="non-linear")

    # The first is above the threshold (2 > 1.5) so the first should be
    # rejected. The threshold is then linearly increased to 1.51. The second
    # candidate is below the threshold (0 < 1.51), so the second should be
    # accepted. The threshold is then exponentially decreased to 1.20. The third
    # candidate is below the theshold (1 < 1.20) and is accepted.
    assert_(not great_deluge(None, One(), Zero(), Two()))
    assert_(great_deluge(None, One(), Zero(), Zero()))
    assert_(great_deluge(None, One(), Zero(), One()))
