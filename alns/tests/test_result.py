from alns import Result
from numpy.testing import assert_equal


def test_result_states():
    result = Result(1, 0)

    assert_equal(result.best_state, 1)
    assert_equal(result.last_state, 0)
