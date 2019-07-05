import sys

import pytest
from numpy.testing import assert_, assert_raises

from alns.Result import Result
from alns.Statistics import Statistics
from alns.exceptions import NotCollectedError
from .states import Sentinel

try:
    from matplotlib.testing.decorators import check_figures_equal
except ImportError:
    def check_figures_equal(*args, **kwargs):       # placeholder
        return check_figures_equal


# HELPERS ----------------------------------------------------------------------


def get_statistics():
    """
    Helper method.
    """
    statistics = Statistics()

    for objective in range(100):
        statistics.collect_objective(objective)

    return statistics


def get_plot(ax, *args, **kwargs):
    """
    Helper method.
    """
    ax.plot(*args, **kwargs)

    ax.set_title("Objective value at each iteration")
    ax.set_ylabel("Objective value")
    ax.set_xlabel("Iteration (#)")


# TESTS ------------------------------------------------------------------------


def test_result_state():
    """
    Tests if the result object correctly returns the passed-in state.
    """
    best = Sentinel()
    result = Result(best)

    assert_(result.best_state is best)


def test_raises_missing_statistics():
    """
    Accessing the statistics object when no statistics have been passed-in
    should raise.
    """
    result = Result(Sentinel())

    with assert_raises(NotCollectedError):
        result.statistics  # pylint: disable=pointless-statement

    result = Result(Sentinel(), Statistics())
    result.statistics  # pylint: disable=pointless-statement


@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="Plot testing is not reliably available for Py3.4")
@check_figures_equal(extensions=['png'])
def test_plot_objectives(fig_test, fig_ref):
    """
    Tests if the ``plot_objectives`` method returns the same figure as a
    reference plot below.
    """
    statistics = get_statistics()
    result = Result(Sentinel(), statistics)

    # Tested plot
    result.plot_objectives(fig_test.subplots())

    # Reference plot
    get_plot(fig_ref.subplots(), statistics.objectives)


@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="Plot testing is not reliably available for Py3.4")
@check_figures_equal(extensions=['png'])
def test_plot_objectives_kwargs(fig_test, fig_ref):
    """
    Tests if the passed-in keyword arguments to ``plot_objectives`` are
    correctly passed to the ``plot`` method.
    """
    statistics = get_statistics()
    result = Result(Sentinel(), statistics)

    kwargs = dict(lw=5, marker='*')

    # Tested plot
    result.plot_objectives(fig_test.subplots(), **kwargs)

    # Reference plot
    get_plot(fig_ref.subplots(), statistics.objectives, **kwargs)


@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="Plot testing is not reliably available for Py3.4")
def test_plot_objectives_default_axes():
    """
    When an axes object is not passed, the ``plot_objectives`` method should
    create a new figure and axes object.
    """
    statistics = get_statistics()

    result = Result(Sentinel(), statistics)
    result.plot_objectives()

    # TODO verify the resulting plot somehow
