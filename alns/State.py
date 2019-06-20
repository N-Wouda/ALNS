from abc import ABC, abstractmethod


class State(ABC):
    """
    State object, which stores a solution via its decision variables. The
    objective value is evaluated via its ``objective()`` member, and should
    return a totally ordered type - e.g. an ``int``, ``float``, or something
    comparable.

    The State class is abstract - you are encouraged to subclass it to suit
    your specific problem.
    """

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
