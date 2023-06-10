from typing import Protocol

import numpy as np


class State(Protocol):
    """
    Protocol for a solution state. Solutions should define an ``objective()``
    member function for evaluation.
    """

    def objective(self) -> float:
        """
        Computes the state's associated objective value.
        """


class ContextualState(Protocol):
    """
    Protocol for a solution state that also provides context. Solutions should
    define an ``objective()`` function as well as a ``get_context()``
    function.
    """

    def objective(self) -> float:
        """
        Computes the state's associated objective value.
        """

    def get_context(self) -> np.ndarray:
        """
        Computes a context vector for the current state
        """
