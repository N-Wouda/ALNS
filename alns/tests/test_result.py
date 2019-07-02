from numpy.testing import assert_

from alns import Result
from .states import Sentinel


def test_result_state():
    """
    Tests if the result object correctly returns the passed-in state.
    """
    best = Sentinel()
    result = Result(best)

    assert_(result.best_state is best)
