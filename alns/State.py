from abc import ABC, abstractmethod


class State(ABC):
    """
    State object, which stores a solution and whose cost can be evaluated
    through its ``objective()`` member function.

    The State class is abstract - you are encouraged to subclass it to suit
    your specific problem.
    """

    @abstractmethod
    def objective(self) -> float:
        """
        Computes the state's associated objective value.
        """
        return NotImplemented
