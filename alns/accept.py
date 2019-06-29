import numpy as np

from .State import State    # pylint: disable=unused-import


def accept(current, candidate, temperature, rnd):
    """
    Implements an annealing schedule acceptance criterion.

    Parameters
    ----------
    current : State
        The current solution state.
    candidate : State
        The candidate solution state.
    temperature : float
        The current temperature.
    rnd : RandomState
        The RandomState from whence to draw random numbers.

    Raises
    ------
    ValueError
        When the temperature is non-positive.

    Returns
    -------
    bool
        True if the candidate solution is to be accepted, False if not.
    """
    if temperature <= 0:
        raise ValueError("Non-positive temperature is not understood.")

    probability = np.exp((current.objective() - candidate.objective())
                         / temperature)

    return probability > rnd.random_sample()
