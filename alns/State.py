from abc import ABC, abstractmethod
from numbers import Number


class State(ABC):
    """
    State object, which stores a solution and whose cost can be evaluated
    through its ``objective()`` member function.

    The State class is abstract - you are encouraged to subclass it to suit
    your specific problem.
    """

    @abstractmethod
    def objective(self) -> Number:
        """
        Computes the state's associated objective value.

        Returns
        -------
        Number
            Some numeric value.
        """
        return NotImplemented
