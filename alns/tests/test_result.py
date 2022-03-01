import numpy as np
import numpy.random as rnd
import pytest
from numpy.testing import assert_, assert_raises

from alns.Result import Result
from alns.Statistics import Statistics
from .states import Sentinel

try:
    from matplotlib.testing.decorators import check_figures_equal
except ImportError:
    def check_figures_equal(*args, **kwargs):  # placeholder
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


def get_objective_plot(ax, data, **kwargs):
    """
    Helper method.
    """
    title = kwargs.pop('title', None)

    if title is None:
        title = "Objective value at each iteration"

    ax.plot(data, **kwargs)
    ax.plot(np.minimum.accumulate(data), **kwargs)

    ax.set_title(title)
    ax.set_ylabel("Objective value")
    ax.set_xlabel("Iteration (#)")

    ax.legend(["Current", "Best"], loc="upper right")


def get_operator_plot(figure, destroy, repair, legend=None, suptitle=None,
                      **kwargs):
    """
    Helper method.
    """

    def _helper(ax, operator_counts, title):
        operator_names = list(operator_counts.keys())

        operator_counts = np.array(list(operator_counts.values()))
        cumulative_counts = operator_counts[:, :len(legend)].cumsum(axis=1)

        ax.set_xlim(right=cumulative_counts[:, -1].max())

        for idx in range(len(legend)):
            widths = operator_counts[:, idx]
            starts = cumulative_counts[:, idx] - widths

            ax.barh(operator_names, widths, left=starts, height=0.5, **kwargs)

            for y, (x, label) in enumerate(zip(starts + widths / 2, widths)):
                ax.text(x, y, str(label), ha='center', va='center')

        ax.set_title(title)
        ax.set_xlabel("Iterations where operator resulted in this outcome (#)")
        ax.set_ylabel("Operator")

    if suptitle is not None:
        figure.suptitle(suptitle)

    if legend is None:
        legend = ["Best", "Better", "Accepted", "Rejected"]

    d_ax, r_ax = figure.subplots(nrows=2)

    _helper(d_ax, destroy, "Destroy operators")
    _helper(r_ax, repair, "Repair operators")

    figure.legend(legend,
                  ncol=len(legend),
                  loc="lower center")


# TESTS ------------------------------------------------------------------------


def test_result_state():
    """
    Tests if the result object correctly returns the passed-in state.
    """
    best = Sentinel()

    assert_(get_result(best).best_state is best)


@pytest.mark.matplotlib
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
def test_plot_objectives_default_axes():
    """
    When an axes object is not passed, the ``plot_objectives`` method should
    create a new figure and axes object.
    """
    result = get_result(Sentinel())
    result.plot_objectives()

    # TODO verify the resulting plot somehow


@pytest.mark.matplotlib
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


@pytest.mark.matplotlib
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
                      suptitle="A random test title")


@pytest.mark.matplotlib
def test_plot_operator_counts_default_figure():
    """
    When a figure object is not passed, the ``plot_operator_counts`` method
    should create new figure and axes objects.
    """
    result = get_result(Sentinel())
    result.plot_operator_counts()

    # TODO verify the resulting plot somehow


@pytest.mark.matplotlib
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


@pytest.mark.matplotlib
@check_figures_equal(extensions=['png'])
def test_plot_operator_counts_legend_length(fig_test, fig_ref):
    """
    Tests if the length of the passed-in legend is used to determine which
    counts to show.
    """
    result = get_result(Sentinel())

    # Tested plot
    result.plot_operator_counts(fig_test, legend=["Best"])

    # Reference plot
    get_operator_plot(fig_ref,
                      result.statistics.destroy_operator_counts,
                      result.statistics.repair_operator_counts,
                      legend=["Best"])
