def update(current: float, step: float, method: str) -> float:
    """
    Updates the passed-in criterion threshold parameter. This is done in one of
    two ways, determined via ``method``. If ``method`` is linear, then ``step``
    is subtracted from the threshold. If ``method`` is exponential, the
    threshold is multiplied by ``step``.

    Parameters
    ----------
    current
        The current criterion threshold.
    step
        The chosen step size.
    method
        The updating method, one of {'linear', 'exponential'}.

    Raises
    ------
    ValueError
        When the method is not understood.

    Returns
    -------
    The new criterion threshold.
    """
    method = method.lower()

    if method == "linear":
        return current - step
    elif method == "exponential":
        return current * step

    raise ValueError("Method `{0}' not understood.".format(method))
