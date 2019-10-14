# EXCEPTIONS -------------------------------------------------------------------


class NotCollectedError(Exception):
    """
    Raised when statistics are accessed from an ALNS result instance, but
    statistics were not collected during iteration.
    """
    pass


# WARNINGS ---------------------------------------------------------------------


class OverwriteWarning(UserWarning):
    """
    Raised when the ALNS instance detects that a new operator has the same name
    as an already existing operator of that type.
    """
    pass
