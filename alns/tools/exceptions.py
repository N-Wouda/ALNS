class NotCollectedError(Exception):
    """
    Raised when statistics are accessed from an ALNS result instance, but
    statistics were not collected during iteration.
    """
    pass
