"""
TA has moved into RRT. These tests remain here for historical reasons.
"""

import numpy.random as rnd
from numpy.testing import assert_

from alns.accept import RecordToRecordTravel
from alns.tests.states import One, Zero


def test_accepts_better():
    threshold_accept = RecordToRecordTravel(1, 0, 0.1, cmp_best=False)
    assert_(threshold_accept(rnd.default_rng(), Zero(), One(), Zero()))


def test_rejects_worse():
    threshold_accept = RecordToRecordTravel(0.5, 0.2, 0.1)

    # This results in a relative worsening of plus one, which is bigger than
    # the threshold (0.5).
    assert_(not threshold_accept(rnd.default_rng(), Zero(), Zero(), One()))


def test_accepts_equal():
    threshold_accept = RecordToRecordTravel(0, 0, 0.1, cmp_best=False)

    # Even at the strictest threshold, this should be accepted since the
    # relative improvement is zero (they are equal).
    assert_(threshold_accept(rnd.default_rng(), Zero(), Zero(), Zero()))


def test_linear_threshold_update():
    threshold_accept = RecordToRecordTravel(5, 0, 1, cmp_best=False)

    # For the first five, the threshold is, resp., 5, 4, 3, 2, 1. The relative
    # worsening is plus one, so this should be accepted (lower or equal to
    # threshold).
    for _ in range(5):
        assert_(threshold_accept(rnd.default_rng(), Zero(), Zero(), One()))

    # Threshold is now zero, so this should no longer be accepted.
    assert_(not threshold_accept(rnd.default_rng(), Zero(), Zero(), One()))


def test_exponential_threshold_update():
    threshold_accept = RecordToRecordTravel(5, 0, 0.1, "exponential", False)

    # The relative worsening is plus one, so this should be accepted initially,
    # as the threshold is 5, resp. 0.5. In the second case, 1 > 0.5, so the
    # second should be rejected.
    assert_(threshold_accept(rnd.default_rng(), Zero(), Zero(), One()))
    assert_(not threshold_accept(rnd.default_rng(), Zero(), Zero(), One()))
