import numpy as np

from .State import State    # pylint: disable=unused-import


def accept(current, candidate, temperature, temperature_decay, rnd):
    """
    Implements an annealing schedule acceptance criterion.

    Parameters
    ----------
    current : State
        The current solution state.
    candidate : State
        The candidate solution state.
    temperature : float
        The initial temperature.
    temperature_decay : float
        The decay parameter, as a number in the unit interval.
    rnd : RandomState
        The RandomState from whence to draw random numbers.

    Raises
    ------
    ValueError
        When the temperature or temperature decay parameters do not meet
        requirements.

    Returns
    -------
    bool
        True if the candidate solution is to be accepted, False if not.

    References
    ----------
    Kirkpatrick, S., Gerlatt, C. D. Jr., and Vecchi, M. P., Optimization by
    Simulated Annealing, *IBM Research Report* RC 9355, 1982.
    """
    if temperature <= 0:
        raise ValueError("Non-positive temperature is not understood.")

    if not (0 < temperature_decay < 1):
        raise ValueError("Temperature decay parameter outside unit interval is"
                         " not understood.")

    probability = np.exp((current.objective() - candidate.objective())
                         / next(_temperature(temperature, temperature_decay)))

    return probability > rnd.random_sample()


def _temperature(temperature, temperature_decay):
    """
    As given in Kirkpatrick et al. (1982).

    Parameters
    ----------
    temperature : float
        The initial temperature.
    temperature_decay : float
        The decay parameter, as a number in the unit interval.
    """
    while True:
        yield temperature
        temperature *= temperature_decay
