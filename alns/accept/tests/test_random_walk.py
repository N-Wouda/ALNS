import numpy.random as rnd
from numpy.testing import assert_

from alns.accept import AlwaysAccept
from alns.tests.states import One, Zero


def test_accepts_better():
    """
    Tests if the always accept method accepts a better solution.
    """
    always_accept = AlwaysAccept()
    assert_(always_accept(rnd.default_rng(), One(), One(), Zero()))


def test_accepts_worse():
    """
    Tests if the always accept method accepts a worse solution.
    """
    always_accept = AlwaysAccept()
    assert_(always_accept(rnd.default_rng(), Zero(), Zero(), One()))


def test_accepts_equal():
    """
    Tests if the always accept method accepts a solution that results in the
    same objective value.
    """
    always_accept = AlwaysAccept()
    assert_(always_accept(rnd.default_rng(), Zero(), Zero(), Zero()))
