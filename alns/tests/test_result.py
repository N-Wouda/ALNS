import os
import sys

import numpy as np
import numpy.random as rnd
import pytest
from numpy.testing import assert_, assert_raises

if "TRAVIS" in os.environ:
    import matplotlib

    # Travis does not understand the default TkAgg back-end, so we have to set
    # this manually. See also https://stackoverflow.com/q/37604289/4316405.
    matplotlib.use('Agg')

from alns.Result import Result
from alns.Statistics import Statistics
from alns.exceptions_warnings import NotCollectedError
from .states import Sentinel

try:
    from matplotlib.testing.decorators import check_figures_equal
except ImportError:
    def check_figures_equal(*args, **kwargs):       # placeholder
        return check_figures_equal


# HELPERS ----------------------------------------------------------------------


def get_result(state):
    """
    Helper method.
    """
    return Result(state, get_statistics())


def get_statistics():
    """
    Helper method.
    """
    statistics = Statistics()

    for objective in range(100):
        statistics.collect_objective(objective)

    # We should make sure these results are reproducible.
    state = rnd.RandomState(1)

    operators = ["test1", "test2", "test3"]

    for _ in range(100):
        operator = state.choice(operators)

        statistics.collect_destroy_operator("d_" + operator, state.randint(4))
        statistics.collect_repair_operator("r_" + operator, state.randint(4))

    return statistics


# TODO revisit image comparison - maybe check against static images instead?


def get_objective_plot(ax, *args, **kwargs):
    """
    Helper method.
    """
    ax.plot(*args, **kwargs)

    ax.set_title("Objective value at each iteration")
    ax.set_ylabel("Objective value")
    ax.set_xlabel("Iteration (#)")


def get_operator_plot(figure, destroy, repair, title=None, **kwargs):
    """
    Helper method.
    """
    def _helper(ax, operator_counts, title, **kwargs):
        operator_names = list(operator_counts.keys())

        operator_counts = np.array(list(operator_counts.values()))
        cumulative_counts = operator_counts.cumsum(axis=1)

        ax.set_xlim(right=np.sum(operator_counts, axis=1).max())

        for idx in range(4):
            widths = operator_counts[:, idx]
            starts = cumulative_counts[:, idx] - widths

            ax.barh(operator_names, widths, left=starts, height=0.5, **kwargs)

            for y, (x, label) in enumerate(zip(starts + widths / 2, widths)):
                ax.text(x, y, str(label), ha='center', va='center')

        ax.set_title(title)
        ax.set_xlabel("Iterations where operator resulted in this outcome (#)")
        ax.set_ylabel("Operator")

    if title is not None:
        figure.suptitle(title)

    d_ax, r_ax = figure.subplots(nrows=2)

    _helper(d_ax, destroy, "Destroy operators", **kwargs)
    _helper(r_ax, repair, "Repair operators", **kwargs)

    figure.legend(["Best", "Better", "Accepted", "Rejected"],
                  ncol=4,
                  loc="lower center")


# TESTS ------------------------------------------------------------------------


def test_result_state():
    """
    Tests if the result object correctly returns the passed-in state.
    """
    best = Sentinel()

    assert_(get_result(best).best_state is best)


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


@pytest.mark.matplotlib
@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="Plot testing is not reliably available for Py3.4")
@check_figures_equal(extensions=['png'])
def test_plot_objectives(fig_test, fig_ref):
    """
    Tests if the ``plot_objectives`` method returns the same figure as a
    reference plot below.
    """
    result = get_result(Sentinel())

    # Tested plot
    result.plot_objectives(fig_test.subplots())

    # Reference plot
    get_objective_plot(fig_ref.subplots(), result.statistics.objectives)


@pytest.mark.matplotlib
@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="Plot testing is not reliably available for Py3.4")
@check_figures_equal(extensions=['png'])
def test_plot_objectives_kwargs(fig_test, fig_ref):
    """
    Tests if the passed-in keyword arguments to ``plot_objectives`` are
    correctly passed to the ``plot`` method.
    """
    result = get_result(Sentinel())
    kwargs = dict(lw=5, marker='*')

    # Tested plot
    result.plot_objectives(fig_test.subplots(), **kwargs)

    # Reference plot
    get_objective_plot(fig_ref.subplots(),
                       result.statistics.objectives,
                       **kwargs)


@pytest.mark.matplotlib
@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="Plot testing is not reliably available for Py3.4")
def test_plot_objectives_default_axes():
    """
    When an axes object is not passed, the ``plot_objectives`` method should
    create a new figure and axes object.
    """
    result = get_result(Sentinel())
    result.plot_objectives()

    # TODO verify the resulting plot somehow


@pytest.mark.matplotlib
@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="Plot testing is not reliably available for Py3.4")
@check_figures_equal(extensions=['png'])
def test_plot_operator_counts(fig_test, fig_ref):
    """
    Tests if the ``plot_operator_counts`` method returns the same figure as a
    reference plot below.
    """
    result = get_result(Sentinel())

    # Tested plot
    result.plot_operator_counts(fig_test)

    # Reference plot
    get_operator_plot(fig_ref,
                      result.statistics.destroy_operator_counts,
                      result.statistics.repair_operator_counts)


def test_plot_operator_counts_raises_legend():
    """
    Tests if the ``plot_operator_counts`` method raises when the passed-in
    legend is of insufficient length.
    """
    result = get_result(Sentinel())

    with assert_raises(ValueError):
        # Legend should be of length four.
        result.plot_operator_counts(legend=["test", "test"])

    # This should work.
    result.plot_operator_counts(legend=["test", "test", "test", "test"])

    # As should longer legend lists - the final values are unused.
    result.plot_operator_counts(legend=["test", "test", "test", "test", "test"])


@pytest.mark.matplotlib
@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="Plot testing is not reliably available for Py3.4")
@check_figures_equal(extensions=['png'])
def test_plot_operator_counts_title(fig_test, fig_ref):
    """
    Tests if ``plot_operator_counts`` sets a plot title correctly.
    """
    result = get_result(Sentinel())

    # Tested plot
    result.plot_operator_counts(fig_test, title="A random test title")

    # Reference plot
    get_operator_plot(fig_ref,
                      result.statistics.destroy_operator_counts,
                      result.statistics.repair_operator_counts,
                      title="A random test title")


@pytest.mark.matplotlib
@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="Plot testing is not reliably available for Py3.4")
def test_plot_operator_counts_default_figure():
    """
    When a figure object is not passed, the ``plot_operator_counts`` method
    should create new figure and axes objects.
    """
    result = get_result(Sentinel())
    result.plot_operator_counts()

    # TODO verify the resulting plot somehow


@pytest.mark.matplotlib
@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="Plot testing is not reliably available for Py3.4")
@check_figures_equal(extensions=['png'])
def test_plot_operator_counts_kwargs(fig_test, fig_ref):
    """
    Tests if the passed-in keyword arguments to ``plot_operator_counts`` are
    correctly passed to the ``barh`` method.
    """
    result = get_result(Sentinel())
    kwargs = dict(alpha=.5, lw=2)

    # Tested plot
    result.plot_operator_counts(fig_test, **kwargs)

    # Reference plot
    get_operator_plot(fig_ref,
                      result.statistics.destroy_operator_counts,
                      result.statistics.repair_operator_counts,
                      **kwargs)
