from alns import Result
from numpy.testing import assert_


def test_result_states():
    """
    Tests if the result object correctly returns the passed-in states.
    """
    best = object()     # two unique sentinel objects - their id's are used
    last = object()     # to differentiate the best and last states below.

    result = Result(best, last)

    assert_(result.best_state is best)
    assert_(result.best_state is not last)

    assert_(result.last_state is last)
    assert_(result.last_state is not best)
