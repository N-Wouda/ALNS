from .CallbackFlag import CallbackFlag


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
        """
        # TODO raise if a callback already exists?
        self._callbacks[CallbackFlag.ON_BEST] = func

    def has_callback(self, flag):
        """
        Determines if a callable has been set for the passed-in flag.

        Parameters
        ----------
        flag : CallbackFlag

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
