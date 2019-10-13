import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes  # pylint: disable=unused-import

from .State import State  # pylint: disable=unused-import
from .Statistics import Statistics  # pylint: disable=unused-import
from .exception_warnings import NotCollectedError


class Result:

    def __init__(self, best, statistics=None):
        """
        Stores ALNS results. An instance of this class is returned once the
        algorithm completes.

        Parameters
        ----------
        best : State
            The best state observed during the entire iteration.
        statistics : Statistics
            Statistics optionally collected during iteration.
        """
        self._best = best
        self._statistics = statistics

    @property
    def best_state(self):
        """
        The best state observed during the entire iteration.

        Returns
        -------
        State
            The associated State object
        """
        return self._best

    @property
    def statistics(self):
        """
        The statistics object populated during iteration.

        Raises
        ------
        NotCollectedError
            When statistics were not collected during iteration. This may be
            remedied by setting the appropriate flag.

        Returns
        -------
        Statistics
            The statistics object.
        """
        if self._statistics is None:
            raise NotCollectedError("Statistics were not collected during "
                                    "iteration.")

        return self._statistics

    def plot_objectives(self, ax=None, **kwargs):
        """
        Plots the collected objective values at each iteration.

        Parameters
        ----------
        ax : Axes
            Optional axes argument. If not passed, a new figure and axes are
            constructed.
        kwargs : dict
            Optional arguments passed to ``ax.plot``.
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.statistics.objectives, **kwargs)

        ax.set_title("Objective value at each iteration")
        ax.set_ylabel("Objective value")
        ax.set_xlabel("Iteration (#)")

        plt.draw_if_interactive()

    def plot_operator_counts(self, axs=None, **kwargs):
        """
        Plots an overview of the destroy and repair operators' performance.

        Parameters
        ----------
        axs : list
            Optional list of length two, containing an Axes object for each of
            the operator types.
        kwargs : dict
            Optional arguments passed to each call of ``ax.plot``.

        Raises
        ------
        ValueError
            When an axs list is passed-in, but not of the appropriate length.
        """
        if axs is None:
            _, (d_ax, r_ax) = plt.subplots(nrows=2)
        elif len(axs) != 2:
            raise ValueError("Expected two axes objects, got {0}."
                             .format(len(axs)))
        else:
            d_ax, r_ax = axs

        # TODO destroy_operator data

        d_ax.set_title("Destroy operators")
        d_ax.set_ylabel("Iterations where operator resulted in"
                        " this outcome (#)")
        d_ax.set_xlabel("Operator")

        # TODO repair_operator data

        r_ax.set_title("Repair operators")
        r_ax.set_ylabel("Iterations where operator resulted in"
                        " this outcome (#)")
        r_ax.set_xlabel("Operator")

        plt.draw_if_interactive()
