import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import NonLinearGreatDeluge
from alns.tests.states import One, Two, VarObj, Zero


@mark.parametrize(
    "alpha, beta, gamma, delta",
    [
        (1, 0.5, 1, 1),  # alpha cannot be <= 1
        (2, 1, 1, 1),  # beta cannot be outside of (0, 1)
        (2, 0.5, 0, 1),  # gamma cannot be <= 0
        (2, 0.5, 1, 0),  # delta cannot be <= 0
    ],
)
def test_raise_invalid_parameters(alpha, beta, gamma, delta):
    with assert_raises(ValueError):
        NonLinearGreatDeluge(alpha, beta, gamma, delta)


@mark.parametrize(
    "alpha,beta,gamma,delta",
    [(2, 0.5, 1, 1), (2, 0.5, 0.5, 0.5), (1.1, 0.01, 0.25, 0.25)],
)
def test_no_raise_valid_parameters(alpha, beta, gamma, delta):
    NonLinearGreatDeluge(alpha, beta, gamma, delta)


@mark.parametrize(
    "alpha,beta,gamma,delta",
    [(2, 0.5, 1, 1), (2, 0.5, 0.5, 0.5), (1.1, 0.01, 0.25, 0.25)],
)
def test_properties(alpha, beta, gamma, delta):
    nlgd = NonLinearGreatDeluge(alpha, beta, gamma, delta)

    assert_equal(nlgd.alpha, alpha)
    assert_equal(nlgd.beta, beta)
    assert_equal(nlgd.gamma, gamma)
    assert_equal(nlgd.delta, delta)


def test_raise_zero_best():
    """
    Test if an error is raised when the initial solution has value 0.
    The initial solution is used to compute the initial threshold. An initial
    threshold with value 0 cannot be updated because the relative gap in the
    update method cannot be computed.
    """
    nlgd = NonLinearGreatDeluge(2, 0.1, 0.01, 1)

    with assert_raises(ValueError):
        nlgd(rnd.default_rng(), Zero(), One(), One())


def test_accepts_below_threshold():
    nlgd = NonLinearGreatDeluge(1.01, 0.5, 0.01, 1)

    # Initial threshold is set at 1.01, hence One should be accepted.
    assert_(nlgd(rnd.default_rng(), One(), Zero(), One()))


def test_rejects_above_threshold():
    nlgd = NonLinearGreatDeluge(1.99, 0.5, 1, 1)

    # Initial threshold is set at 1.99 and the current solution is Zero,
    # hence Two should be rejected.
    assert_(not nlgd(rnd.default_rng(), One(), Zero(), Two()))


def test_rejects_equal_threshold():
    nlgd = NonLinearGreatDeluge(2, 0.5, 1, 1)

    # Initial threshold is set at 2 and the current solution is Zero,
    # hence Two should be rejected.
    assert_(not nlgd(rnd.default_rng(), One(), Zero(), Two()))


def test_accepts_improving_current():
    nlgd = NonLinearGreatDeluge(2, 0.5, 1, 1)

    # Candidate is not below the threshold (2 == 2) but does improve the
    # current solution (2 < 3), hence candidate should be accepted.
    assert_(nlgd(rnd.default_rng(), One(), VarObj(3), Two()))


def test_evaluate_consecutive_solutions():
    """
    Test if NLGD correctly accepts and rejects consecutive solutions.
    """
    nlgd = NonLinearGreatDeluge(1.5, 0.1, 0.02, 2)

    # The first is above the threshold (2 > 1.5) so the first should be
    # rejected. The threshold is then linearly increased to 1.51. The second
    # candidate is below the threshold (0 < 1.51), so the second should be
    # accepted. The threshold is then exponentially decreased to 1.20. The
    # third candidate is below the theshold (1 < 1.20) and is accepted.
    assert_(not nlgd(rnd.default_rng(), One(), Zero(), Two()))
    assert_(nlgd(rnd.default_rng(), One(), Zero(), Zero()))
    assert_(nlgd(rnd.default_rng(), One(), Zero(), One()))
