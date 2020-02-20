import warnings

from .CallbackFlag import CallbackFlag
from .exceptions_warnings import OverwriteWarning


class CallbackMixin:

    def __init__(self):
        """
        Callback mix-in for ALNS. This allows for some flexibility by having
        ALNS call custom functions whenever a special event happens.
        """
        self._callbacks = {}

    def on_best(self, func):
        """
        Sets a callback function to be called when ALNS finds a new global best
        solution state.

        Parameters
        ----------
        func : callable
            A function that should take a solution State as its first parameter,
            and a numpy RandomState as its second (cf. the operator signature).
            It should return a (new) solution State.

        Warns
        -----
        OverwriteWarning
            When a callback has already been set for the ON_BEST flag.
        """
        self._set_callback(CallbackFlag.ON_BEST, func)

    def has_callback(self, flag):
        """
        Determines if a callable has been set for the passed-in flag.

        Parameters
        ----------
        flag : CallbackFlag

        Returns
        -------
        bool
            True if a callable is set, False otherwise.
        """
        return flag in self._callbacks

    def callback(self, flag):
        """
        Returns the callback for the passed-in flag, assuming it exists.

        Parameters
        ----------
        flag : CallbackFlag
            The callback flag for which to retrieve a callback.

        Returns
        -------
        callable
            Callback for the passed-in flag.
        """
        return self._callbacks[flag]

    def _set_callback(self, flag, func):
        """
        Sets the passed-in callback func for the passed-in flag. Warns if this
        would overwrite an existing callback.
        """
        if self.has_callback(flag):
            warnings.warn("A callback function has already been set for the"
                          " `{0}' flag. This callback will now be replaced by"
                          " the newly passed-in callback.".format(flag),
                          OverwriteWarning)

        self._callbacks[flag] = func
