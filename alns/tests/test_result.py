from numpy.testing import assert_

from alns import Result
from .states import Sentinel


def test_result_states():
    """
    Tests if the result object correctly returns the passed-in states.
    """
    best = Sentinel()     # two unique sentinel objects - their ids are used
    last = Sentinel()     # to differentiate the best and last states below.

    assert_(best is not last)                               # sanity check

    result = Result(best, last)

    assert_(result.best_state is best)
    assert_(result.best_state is not last)

    assert_(result.last_state is last)
    assert_(result.last_state is not best)
