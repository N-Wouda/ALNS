from abc import ABC, abstractmethod


class State(ABC):
    """
    State object, which stores a solution and whose cost can be evaluated
    through its ``objective()`` member function.
    """

    @abstractmethod
    def objective(self) -> float:
        """
        Computes the state's associated objective value.
        """
        return NotImplemented
