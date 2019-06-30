from .State import State  # pylint: disable=unused-import


class Result:

    def __init__(self, best, last):
        """
        Stores ALNS results. An instance of this class is returned once the
        algorithm completes.

        Parameters
        ----------
        best : State
            The best state observed during the entire iteration.
        last : State
            The last accepted state before the algorithm terminated.
        """
        self._best = best
        self._last = last

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
    def last_state(self):
        """
        The last accepted state before the algorithm terminated.

        Returns
        -------
        State
            The associated State object
        """
        return self._last
