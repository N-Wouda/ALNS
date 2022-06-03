from alns import State


class Zero(State):
    """
    Testable state with objective zero.
    """

    def objective(self):
        return 0


class One(State):
    """
    Testable state with objective one.
    """

    def objective(self):
        return 1


class Two(State):
    """
    Testable state with objective two.
    """

    def objective(self):
        return 2


class Three(State):
    """
    Testable state with objective three.
    """

    def objective(self):
        return 3


class Sentinel(Zero):
    """
    Placeholder state.
    """
