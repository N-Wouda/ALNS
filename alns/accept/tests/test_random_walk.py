import numpy.random as rnd
from numpy.testing import assert_

from alns.accept import RandomWalk
from alns.tests.states import One, Zero


def test_accepts_better():
    """
    Tests if the random walk method accepts a better solution.
    """
    random_walk = RandomWalk()
    assert_(random_walk(rnd.RandomState(), One(), One(), Zero()))


def test_accepts_worse():
    """
    Tests if the random walk method accepts a worse solution.
    """
    random_walk = RandomWalk()
    assert_(random_walk(rnd.RandomState(), Zero(), Zero(), One()))


def test_accepts_equal():
    """
    Tests if the random walk method accepts a solution that results in the
    same objective value.
    """
    random_walk = RandomWalk()
    assert_(random_walk(rnd.RandomState(), Zero(), Zero(), Zero()))
