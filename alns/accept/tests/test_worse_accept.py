from unittest.mock import Mock

import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import WorseAccept
from alns.tests.states import One, Two, Zero


@mark.parametrize(
    "start, end, step, method",
    [
        (-1, 1, 0.1, "linear"),  # start prob cannot be < 0
        (2, 1, 0.1, "linear"),  # start prob cannot be > 1
        (1, -1, 0.1, "linear"),  # end prob cannot be < 0
        (1, 2, 0.1, "linear"),  # end prob cannot be > 1
        (1, 0, -0.1, "linear"),  # updating step cannot be < 0
        (0.5, 0.6, 0.1, "linear"),  # start prob cannot be < end prob
        (1, 0.5, 2, "exponential"),  # step cannot be > 1 with exponential
    ],
)
def test_raises_invalid_parameters(start, end, step, method):
    with assert_raises(ValueError):
        WorseAccept(start, end, step, method)


@mark.parametrize(
    "start, end, step, method",
    [
        (1, 1, 1, "linear"),  # one start and end prob
        (0, 0, 1, "linear"),  # zero start and end prob
        (0.5, 0.5, 1, "linear"),  # equal start and end prob
        (0.05, 0.01, 0.001, "linear"),  # regular start and end prob
        (1, 0.5, 1, "exponential"),  # boundary step exponential
    ],
)
def test_no_raise_valid_parameters(start, end, step, method):
    WorseAccept(start, end, step, method)


@mark.parametrize(
    "start,end,step,method",
    [
        (1, 0, 1, "linear"),
        (0.9, 0.0, 0.1, "linear"),
        (0.5, 0.5, 0.1, "linear"),
        (0, 0, 0, "exponential"),
        (1, 0, 0.9999, "exponential"),
    ],
)
def test_properties(start, end, step, method):
    """
    Tests if the properties are correctly set.
    """
    worse_accept = WorseAccept(start, end, step, method)

    assert_equal(worse_accept.start_prob, start)
    assert_equal(worse_accept.end_prob, end)
    assert_equal(worse_accept.step, step)
    assert_equal(worse_accept.method, method)


def test_zero_prob_accepts_better():
    """
    Tests if WA with a zero start probability accepts better solutions.
    """
    rnd_vals = [1]
    rng = Mock(spec_set=rnd.RandomState, random=lambda: rnd_vals.pop(0))
    worse_accept = WorseAccept(0, 0, 0.1)

    assert_(worse_accept(rng, Zero(), One(), Zero()))
    assert_(worse_accept(rng, Zero(), Two(), Zero()))


def test_zero_prob_never_accept_worse():
    """
    Tests if WA with a zero start probability does not accept worse solutions.
    """
    worse_accept = WorseAccept(0, 0, 0, "linear")

    assert_(not worse_accept(rnd.RandomState(), Zero(), One(), One()))
    assert_(not worse_accept(rnd.RandomState(), Zero(), Zero(), One()))


def test_one_prob_always_accept():
    """
    Tests if WA with a fixed probability of 1 leads to always accepting
    solutions.
    """
    worse_accept = WorseAccept(1, 0, 0, "linear")

    for _ in range(100):
        assert_(worse_accept(rnd.RandomState(), Zero(), Zero(), One()))


def test_linear_consecutive_solutions():
    """
    Test if WA with linear updating method correctly accepts and rejects
    consecutive solutions.
    """
    rnd_vals = [0.9, 0.8, 0.7, 0.6, 0.5, 1]
    rng = Mock(spec_set=rnd.RandomState, random=lambda: rnd_vals.pop(0))
    worse_accept = WorseAccept(1, 0, 0.1, "linear")

    # For the first five, the probability is, resp., 1, 0.9, 0.8, 0.7, 0.6
    # The random draw is, resp., 0.9, 0.8, 0.7, 0.6, 0.5 so the worsening
    # solution is still accepted.
    for _ in range(5):
        assert_(worse_accept(rng, Zero(), Zero(), One()))

    # The probability is now 0.5 and the draw is 1, so reject.
    assert_(not worse_accept(rng, Zero(), Zero(), One()))


def test_exponential_consecutive_solutions():
    """
    Test if WA with exponential updating method correctly accepts and rejects
    consecutive solutions.
    """
    rnd_vals = [0.5, 0.25, 0.125, 1]
    rng = Mock(spec_set=rnd.RandomState, random=lambda: rnd_vals.pop(0))
    worse_accept = WorseAccept(1, 0, 0.5, "exponential")

    # For the first three, the probability is, resp., 1, 0.5, 0.25
    # The random draw is, resp., 0.5, 0.25, 0.125, so the worsening
    # solution is still accepted.
    for _ in range(3):
        assert_(worse_accept(rng, Zero(), Zero(), One()))

    # The probability is now 0.5 and the draw is 1, so reject.
    assert_(not worse_accept(rng, Zero(), Zero(), One()))
