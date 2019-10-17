import numpy as np


def select_operator(operators, weights, rnd_state):
    """
    Selects an operator from the list of operators, using a distribution
    inferred from the given weights.

    Parameters
    ----------
    operators : array_like
        The list of operators.
    weights : array_like
        The operator weights.
    rnd_state : rnd.RandomState
        Random state to draw the choice from.

    Returns
    -------
    int
        Index into the operator array of the selected method.
    """
    return rnd_state.choice(np.arange(0, len(operators)),
                            p=weights / np.sum(weights))
