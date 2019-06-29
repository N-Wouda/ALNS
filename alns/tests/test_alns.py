from numpy.testing import assert_equal

from alns import ALNS
from .states import One, Zero


def test_trivial_example():
    """
    This tests the ALNS algorithm on a trivial example, where the initial
    solution is zero, and any other operator returns one.
    """
    alns = ALNS()

    alns.add_repair_operator(lambda state, rnd: Zero())
    alns.add_destroy_operator(lambda state, rnd: Zero())

    result = alns.iterate(One(), [1, 1, 1, 1], .5, 100)

    assert_equal(result.best_state.objective(), 0)
    assert_equal(result.last_state.objective(), 0)

# TODO more sophisticated tests
