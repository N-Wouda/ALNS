def update(current, step, method):
    """
    Updates the passed-in criterion threshold parameter. This is done in one of
    two ways, determined via ``method``. If ``method`` is linear, then ``step``
    is subtracted from the threshold. If ``method`` is exponential, the
    threshold is multiplied by ``step``.

    Parameters
    ----------
    current : float
        The current criterion threshold.
    step : float
        The chosen step size.
    method : str
        The updating method, one of {'linear', 'exponential'}.

    Returns
    -------
    float
        The new criterion threshold.
    """
    method = method.lower()

    if method == "linear":
        return current - step
    elif method == "exponential":
        return current * step

    raise ValueError("Method `{0}' not understood.".format(method))
