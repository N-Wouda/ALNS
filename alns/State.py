from abc import ABC, abstractmethod


class State(ABC):
    """
    State object, which stores a solution via its decision variables. The
    objective value is evaluated via its ``objective()`` member, and should
    return a numeric type - e.g. an ``int``, ``float``, or comparable.

    The State class is abstract - you are encouraged to subclass it to suit
    your specific problem.
    """

    @abstractmethod
    def objective(self):
        """
        Computes the state's associated objective value.

        Returns
        -------
        float
            Some numeric value, e.g. an ``int`` or ``float``.
        """
        return NotImplemented
