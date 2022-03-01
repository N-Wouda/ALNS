from abc import ABC, abstractmethod


class State(ABC):
    """
    State object, which stores a solution via its decision variables. The
    objective value is evaluated via its ``objective()`` member.

    The State class is abstract - you are encouraged to subclass it to suit
    your specific problem.
    """

    @abstractmethod
    def objective(self) -> float:
        """
        Computes the state's associated objective value.

        Returns
        -------
        Some numeric value, e.g. an ``int`` or ``float``.
        """
        return NotImplemented
