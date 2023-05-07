from typing import Protocol


class State(Protocol):
    """
    Protocol for a solution state. Solutions should define an ``objective()``
    member function for evaluation.
    """

    def objective(self) -> float:
        """
        Computes the state's associated objective value.
        """
