class NotCollectedError(Exception):
    """
    Raised when statistics are accessed from an ALNS Result instance, but those
    were not collected during iteration.
    """
    pass
