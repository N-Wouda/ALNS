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
        ...  # pragma: no cover


class ContextualState(State, Protocol):
    """
    Protocol for a solution state that also provides context. Solutions should
    define ``objective()`` and ``get_context()`` methods.
    """

    def get_context(self) -> np.ndarray:
        """
        Computes a context vector for the current state.
        """
        ...  # pragma: no cover
