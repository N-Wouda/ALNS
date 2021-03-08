import numpy.random as rnd
from numpy.testing import assert_

from alns.criteria import HillClimbing
from alns.tests.states import Zero, One


def test_accepts_better():
    """
    Tests if the hill climbing method accepts a better solution.
    """
    hill_climbing = HillClimbing()
    assert_(hill_climbing(rnd.RandomState(), One(), One(), Zero()))


def test_rejects_worse():
    """
    Tests if the hill climbing method accepts a worse solution.
    """
    hill_climbing = HillClimbing()
    assert_(not hill_climbing(rnd.RandomState(), Zero(), Zero(), One()))


def test_accepts_equal():
    """
    Tests if the hill climbing method accepts a solution that results in the
    same objective value.
    """
    hill_climbing = HillClimbing()
    assert_(hill_climbing(rnd.RandomState(), Zero(), Zero(), Zero()))
