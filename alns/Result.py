from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes, Figure

from alns.State import State
from alns.Statistics import Statistics


class Result:
    """
    Stores ALNS results. An instance of this class is returned once the
    algorithm completes.

    Parameters
    ----------
    best
        The best state observed during the entire iteration.
    statistics
        Statistics collected during iteration.
    """

    def __init__(self, best: State, statistics: Statistics):
        self._best = best
        self._statistics = statistics

    @property
    def best_state(self) -> State:
        """
        The best state observed during the entire iteration.
        """
        return self._best

    @property
    def statistics(self) -> Statistics:
        """
        The statistics object populated during iteration.
        """
        return self._statistics

    def plot_objectives(
        self,
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ):
        """
        Plots the collected objective values at each iteration.

        Parameters
        ----------
        ax
            Optional axes argument. If not passed, a new figure and axes are
            constructed.
        title
            Optional title argument. When not passed, a default is set.
        kwargs
            Optional arguments passed to ``ax.plot``.
        """
        if ax is None:
            _, ax = plt.subplots()

        if title is None:
            title = "Objective value at each iteration"

        # First call is current solution objectives (at each iteration), second
        # call is the best solution found so far (as a running minimum).
        ax.plot(self.statistics.objectives, **kwargs)
        ax.plot(np.minimum.accumulate(self.statistics.objectives), **kwargs)

        ax.set_title(title)
        ax.set_ylabel("Objective value")
        ax.set_xlabel("Iteration (#)")

        ax.legend(["Current", "Best"], loc="upper right")

        plt.draw_if_interactive()

    def plot_operator_counts(
        self,
        fig: Optional[Figure] = None,
        title: Optional[str] = None,
        legend: Optional[List[str]] = None,
        **kwargs: Dict[str, Any]
    ):
        """
        Plots an overview of the destroy and repair operators' performance.

        Parameters
        ----------
        fig
            Optional figure. If not passed, a new figure is constructed, and
            some default margins are set.
        title
            Optional figure title. When not passed, no title is set.
        legend
            Optional legend entries. When passed, this should be a list of at
            most four strings. The first string describes the number of times
            a best solution was found, the second a better, the third a solution
            was accepted but did not improve upon the current or global best,
            and the fourth the number of times a solution was rejected. If less
            than four strings are passed, only the first len(legend) count types
            are plotted. When not passed, a sensible default is set and all
            counts are shown.
        kwargs
            Optional arguments passed to each call of ``ax.barh``.
        """
        if fig is None:
            fig, (d_ax, r_ax) = plt.subplots(nrows=2)
            fig.subplots_adjust(hspace=0.7, bottom=0.2)
        else:
            d_ax, r_ax = fig.subplots(nrows=2)

        if title is not None:
            fig.suptitle(title)

        if legend is None:
            legend = ["Best", "Better", "Accepted", "Rejected"]

        self._plot_op_counts(
            d_ax,
            self.statistics.destroy_operator_counts,
            "Destroy operators",
            min(len(legend), 4),
            **kwargs
        )

        self._plot_op_counts(
            r_ax,
            self.statistics.repair_operator_counts,
            "Repair operators",
            min(len(legend), 4),
            **kwargs
        )

        fig.legend(legend[:4], ncol=len(legend), loc="lower center")

        plt.draw_if_interactive()

    @staticmethod
    def _plot_op_counts(ax, operator_counts, title, num_types, **kwargs):
        """
        Internal helper that plots the passed-in operator_counts on the given
        ax object.

        Note
        ----
        This code takes loosely after an example from the matplotlib gallery
        titled "Discrete distribution as horizontal bar chart".
        """
        operator_names = list(operator_counts.keys())

        operator_counts = np.array(list(operator_counts.values()))
        cumulative_counts = operator_counts[:, :num_types].cumsum(axis=1)

        ax.set_xlim(right=cumulative_counts[:, -1].max())

        for idx in range(num_types):
            widths = operator_counts[:, idx]
            starts = cumulative_counts[:, idx] - widths

            ax.barh(operator_names, widths, left=starts, height=0.5, **kwargs)

            for y, (x, label) in enumerate(zip(starts + widths / 2, widths)):
                ax.text(x, y, str(label), ha="center", va="center")

        ax.set_title(title)
        ax.set_xlabel("Iterations where operator resulted in this outcome (#)")
        ax.set_ylabel("Operator")
