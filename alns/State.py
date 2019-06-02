from abc import ABC, abstractmethod


class State(ABC):

    @classmethod
    @abstractmethod
    def from_state(cls, state):
        """
        Constructs a new state from the passed-in state. This may be used to
        ensure solution steps do not overwrite one another for more difficult
        problems.

        Parameters
        ----------
        state : State
            The current state

        Returns
        -------
        State
            The constructed, new state.
        """
        return NotImplemented

    @property
    @abstractmethod
    def objective(self):
        """
        Computes the state's associated objective value.

        Returns
        -------
        Any
            The actual value and type are unimportant, so long as they admit
            comparison and equality operators.
        """
        return NotImplemented
