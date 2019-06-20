import numpy.random as rnd
from numpy.testing import assert_

from alns.accept import accept
from .states import Zero, One


def test_accept_random_solutions():
    """
    Checks if the ``accept`` method correctly decides in two known cases for
    a fixed seed.
    """
    state = rnd.RandomState(0)

    # Using the above seed, the first two random numbers are 0.55 and .72,
    # respectively. The acceptance probability is 0.61, so the first should be
    # accepted (0.61 > 0.55), but the second should not (0.61 < 0.72).
    assert_(accept(One(), Zero(), 2, state))
    assert_(not accept(One(), Zero(), 2, state))


def test_always_accept_better_solutions():
    """
    The temperature-based acceptance criterion is such that it always accepts
    better solutions than the current one.
    """
    state = rnd.RandomState()

    for _ in range(100):
        assert_(accept(Zero(), One(), 2, state))
