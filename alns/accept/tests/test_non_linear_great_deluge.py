import numpy as np
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)
from pytest import mark

from alns.accept import NonLinearGreatDeluge
from alns.tests.states import One, Zero, Two, VarObj


def test_raise_invalid_parameters():
    with assert_raises(ValueError):
        NonLinearGreatDeluge(1, 0.5, 1, 1)  # alpha cannot be <= 1

    with assert_raises(ValueError):
        NonLinearGreatDeluge(2, 1, 1, 1)  # beta cannot be outside of (0, 1)

    with assert_raises(ValueError):
        NonLinearGreatDeluge(2, 0.5, 0, 1)  # gamma cannot be <= 0

    with assert_raises(ValueError):
        NonLinearGreatDeluge(2, 0.5, 1, 0)  # delta cannot be <= 0


def test_does_not_raise():
    NonLinearGreatDeluge(2, 0.5, 1, 1)
    NonLinearGreatDeluge(2, 0.5, 0.5, 0.5)


@mark.parametrize("alpha", np.arange(1.1, 10, 1))
def test_alpha(alpha):
    nlgd = NonLinearGreatDeluge(alpha, 0.1, 1, 1)
    assert_equal(nlgd.alpha, alpha)


@mark.parametrize("beta", np.arange(0.1, 1, 0.1))
def test_beta(beta):
    nlgd = NonLinearGreatDeluge(2, beta, 1, 1)
    assert_equal(nlgd.beta, beta)


@mark.parametrize("gamma", np.arange(0.1, 1, 1))
def test_gamma(gamma):
    nlgd = NonLinearGreatDeluge(2, 0.1, gamma, 1)
    assert_equal(nlgd.gamma, gamma)


@mark.parametrize("delta", np.arange(0.1, 1, 0.1))
def test_delta(delta):
    nlgd = NonLinearGreatDeluge(2, 0.1, 1, delta)
    assert_equal(nlgd.delta, delta)


def test_raise_zero_best():
    """
    Test if an error is raised when the initial solution has value 0
    using NLGD.

    Reason: the relative gap cannot be computed with zero threshold.
    """
    nlgd = NonLinearGreatDeluge(2, 0.1, 0.01, 1)

    with assert_raises(ValueError):
        nlgd(None, Zero(), One(), One())


def test_accepts_below_threshold():
    nlgd = NonLinearGreatDeluge(1.01, 0.5, 0.01, 1)

    # Initial threshold is set at 1.01, hence One should be accepted
    assert_(nlgd(None, One(), Zero(), One()))


def test_rejects_above_threshold():
    nlgd = NonLinearGreatDeluge(2, 0.5, 1, 1)

    # Initial threshold is set at 2 and the current solution is Zero,
    # hence Two should be rejected
    assert_(not nlgd(None, One(), Zero(), Two()))


def test_accepts_improving_current():
    nlgd = NonLinearGreatDeluge(2, 0.5, 1, 1)

    # Candidate is does not improve the threshold (2 == 2) but does improve the
    # current solution value (2 < 3), hence candidate should be accepted
    assert_(nlgd(None, One(), VarObj(3), Two()))


def test_evaluate_consecutive_solutions():
    """
    Test if NLGD correctly accepts and rejects consecutive solutions.
    """
    nlgd = NonLinearGreatDeluge(1.5, 0.1, 0.02, 2)

    # The first is above the threshold (2 > 1.5) so the first should be
    # rejected. The threshold is then linearly increased to 1.51. The second
    # candidate is below the threshold (0 < 1.51), so the second should be
    # accepted. The threshold is then exponentially decreased to 1.20. The third
    # candidate is below the theshold (1 < 1.20) and is accepted.
    assert_(not nlgd(None, One(), Zero(), Two()))
    assert_(nlgd(None, One(), Zero(), Zero()))
    assert_(nlgd(None, One(), Zero(), One()))
