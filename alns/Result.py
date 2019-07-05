import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes  # pylint: disable=unused-import

from .State import State  # pylint: disable=unused-import
from .Statistics import Statistics  # pylint: disable=unused-import
from .exceptions import NotCollectedError


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
