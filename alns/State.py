from abc import ABC, abstractmethod


class State(ABC):

    @abstractmethod
    def copy(self):
        """
        Constructs a new state from this state. This may be used to ensure
        solution steps do not overwrite one another for more difficult
        problems that track many decision variables.

        Returns
        -------
        State
            A newly constructed state, identical to this state.
        """
        return NotImplemented

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
