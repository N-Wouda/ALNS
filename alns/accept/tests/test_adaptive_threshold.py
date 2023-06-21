import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept.AdaptiveThreshold import AdaptiveThreshold


class MockCandidate:
    def __init__(self, objective):
        self._objective = objective

    def objective(self):
        return self._objective


@mark.parametrize(
    "heta, gamma",
    [
        (1.2, 3),  # heta cannot be < 0 or > 1
        (0.5, -2),  # gamma cannot be < 0
    ],
)
def test_raise_invalid_parameters(heta, gamma):
    with assert_raises(ValueError):
        AdaptiveThreshold(heta, gamma)


@mark.parametrize("heta, gamma", [(1, 3), (0.4, 4)])
def test_no_raise_valid_parameters(heta, gamma):
    AdaptiveThreshold(heta, gamma)


@mark.parametrize("heta", np.arange(0, 1, -1))
def test_heta(heta):
    adaptive_threshold = AdaptiveThreshold(heta, 3)
    assert_equal(adaptive_threshold.heta, heta)


@mark.parametrize("gamma", np.arange(2, 11, -1))
def test_gamma(gamma):
    adaptive_threshold = AdaptiveThreshold(0.5, gamma)
    assert_equal(adaptive_threshold.gamma, gamma)


def test_accepts_below_threshold():
    adaptive_threshold = AdaptiveThreshold(0.5, 4)
    adaptive_threshold(MockCandidate(7100))
    adaptive_threshold(MockCandidate(7300))
    result = adaptive_threshold(MockCandidate(7125))

    # The threshold is set at 7100 + 0.5 * (7175 - 7100) = 7137.5
    assert_(result)


def test_rejects_above_threshold():
    adaptive_threshold = AdaptiveThreshold(0.5, 4)
    adaptive_threshold(MockCandidate(7100))
    adaptive_threshold(MockCandidate(7300))
    result = adaptive_threshold(MockCandidate(7225))

    # The threshold is set at 7100 + 0.5 * (7208.33 - 7100) = 7154.17
    assert_(not result)


def test_accepts_equal_threshold():
    adaptive_threshold = AdaptiveThreshold(0.5, 4)
    adaptive_threshold(MockCandidate(7100))
    adaptive_threshold(MockCandidate(7200))
    result = adaptive_threshold(MockCandidate(7120))

    # The threshold is set at 7100 + 0.5 * (7140 - 7100) = 7120
    assert_(result)


def test_accepts_over_gamma_candidates():
    adaptive_threshold = AdaptiveThreshold(0.2, 3)
    adaptive_threshold(MockCandidate(7100))
    adaptive_threshold(MockCandidate(7200))
    adaptive_threshold(MockCandidate(7200))
    result = adaptive_threshold(MockCandidate(7000))

    # The threshold is set at 7000 + 0.2 * (7133.33 - 7000) = 7013.33
    assert_(result)


def test_rejects_over_gamma_candidates():
    adaptive_threshold = AdaptiveThreshold(0.2, 3)
    adaptive_threshold(MockCandidate(7100))
    adaptive_threshold(MockCandidate(7200))
    adaptive_threshold(MockCandidate(7200))
    adaptive_threshold(MockCandidate(7000))
    result = adaptive_threshold(MockCandidate(7100))

    # The threshold is set at 7000 + 0.2 * (7100 - 7000) = 7020
    assert_(not result)


def test_evaluate_consecutive_solutions():
    """
    Test if AT correctly accepts and rejects consecutive solutions.
    """
    adaptive_threshold = AdaptiveThreshold(0.5, 4)

    result = adaptive_threshold(MockCandidate(7100))
    # The threshold is set at 7100, hence the solution is accepted
    assert_(result)

    result = adaptive_threshold(MockCandidate(7200))
    # The threshold is set at 7125, hence the solution is accepted
    assert_(not result)

    result = adaptive_threshold(MockCandidate(7120))
    # The threshold is set at 7120, hence the solution is accepted
    assert_(result)
