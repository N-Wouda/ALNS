from typing import Protocol


class State(Protocol):
    """
    State object, which stores a solution and whose cost can be evaluated
    through its ``objective()`` member function.
    """

    def objective(self) -> float:
        """
        Computes the state's associated objective value.
        """
        pass
