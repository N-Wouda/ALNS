from numpy.testing import assert_, assert_raises

from alns import Result
from alns.Statistics import Statistics
from alns.exceptions import NotCollectedError
from .states import Sentinel


def test_result_state():
    """
    Tests if the result object correctly returns the passed-in state.
    """
    best = Sentinel()
    result = Result(best)

    assert_(result.best_state is best)


def test_raises_no_statistics():
    """
    Accessing the statistics object when no statistics have been passed-in
    should raise
    """
    result = Result(Sentinel())

    with assert_raises(NotCollectedError):
        result.statistics

    result = Result(Sentinel(), Statistics())
    result.statistics                               # this should be fine


# TODO test plot_objectives
