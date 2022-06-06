import numpy as np
import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import WorseAccept
from alns.tests.states import Zero, One, Two


class MockRNG:
    """
    Mock Random Number Generator.
    """

    def __init__(self, values):
        self.values = values

    def random(self):
        return self.values.pop(0)


@mark.parametrize(
    "start, end, step",
    [
        (-1, 1, 0.1),  # start prob cannot be negative or smaller than end prob
        (2, 1, 0.1),  # start prob cannot be larger than 1
        (1, -1, 0.1),  # end prob cannot be negative
        (1, 2, 0.1),  # end prob cannot be larger than 1 or start prob
        (1, 0, -0.1),  # updating step cannot be negative
    ],
)
def test_raises_invalid_parameters(start, end, step):
    """
    Worse accept does not work with non-probabilistic start and end probability
    so those should not be accepted. Negative step values should also not be
    accepted.
    """
    with assert_raises(ValueError):
        WorseAccept(start, end, step)


def test_raises_explosive_step():
    """
    For exponential updating, the step parameter must not be bigger than one,
    as that would result in an explosive probability.
    """
    with assert_raises(ValueError):
        WorseAccept(1, 0.5, 2, "exponential")

    WorseAccept(1, 0.5, 1, "exponential")  # boundary should be fine


def test_prob_boundary():
    """
    The boundary case for the end probability parameter is at zero, which should
    be accepted.
    """
    WorseAccept(1, 0, 1)


def test_raises_start_smaller_than_end():
    """
    The initial probability at the start should be bigger (or equal) to the end
    probability.
    """
    with assert_raises(ValueError):
        WorseAccept(0, 1, 1)

    WorseAccept(1, 1, 1)  # should not raise for equality


@mark.parametrize("step", np.arange(2, 0, -0.1))
def test_step(step):
    """
    Tests if the step parameter is correctly set.
    """
    assert_equal(WorseAccept(1, 1, step).step, step)


@mark.parametrize("start", np.arange(1, 0, -0.1))
def test_start_prob(start):
    """
    Tests if the start_prob parameter is correctly set.
    """
    assert_equal(WorseAccept(start, 0, 1).start_prob, start)


@mark.parametrize("end", np.arange(1, 0, -0.1))
def test_end_prob(end):
    """
    Tests if the end_prob parameter is correctly set.
    """
    assert_equal(WorseAccept(1, end, 1).end_prob, end)


def test_zero_prob_accepts_better():
    """
    Tests if WA with a zero start probability accepts better solutions.
    """
    rng = MockRNG([1])
    worse_accept = WorseAccept(0, 0, 0.1)
    assert_(worse_accept(rng, None, One(), Zero()))
    assert_(worse_accept(rng, None, Two(), Zero()))


def test_zero_prob_never_accept_worse():
    """
    Tests if WA with a zero start probability does not accept worse solutions.
    """
    worse_accept = WorseAccept(0, 0, 0, "linear")

    assert_(not worse_accept(rnd.RandomState(), None, One(), One()))
    assert_(not worse_accept(rnd.RandomState(), None, Zero(), One()))


def test_one_prob_always_accept():
    """
    Tests if WA with a fixed probability of 1 leads to always accepting
    solutions.
    """
    worse_accept = WorseAccept(1, 0, 0, "linear")

    for _ in range(100):
        assert_(worse_accept(rnd.RandomState(), None, Zero(), One()))


def test_linear_consecutive_solutions():
    """
    Test if WA with linear updating method correctly accepts and rejects
    consecutive solutions.
    """
    rng = MockRNG([0.9, 0.8, 0.7, 0.6, 0.5, 1])
    worse_accept = WorseAccept(1, 0, 0.1, "linear")

    # For the first five, the probability is, resp., 1, 0.9, 0.8, 0.7, 0.6
    # The random draw is, resp., 0.9, 0.8, 0.7, 0.6, 0.5 so the worsening
    # solution is still accepted.
    for _ in range(5):
        assert_(worse_accept(rng, None, Zero(), One()))

    # The probability is now 0.5 and the draw is 1, so reject.
    assert_(not worse_accept(rng, None, Zero(), One()))


def test_exponential_consecutive_solutions():
    """
    Test if WA with exponential updating method correctly accepts and rejects
    consecutive solutions.
    """
    rng = MockRNG([0.5, 0.25, 0.125, 1])
    worse_accept = WorseAccept(1, 0, 0.5, "exponential")

    # For the first three, the probability is, resp., 1, 0.5, 0.25
    # The random draw is, resp., 0.5, 0.25, 0.125, so the worsening
    # solution is still accepted.
    for _ in range(3):
        assert_(worse_accept(rng, None, Zero(), One()))

    # The probability is now 0.5 and the draw is 1, so reject.
    assert_(not worse_accept(rng, None, Zero(), One()))
