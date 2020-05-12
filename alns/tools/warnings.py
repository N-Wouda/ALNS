class OverwriteWarning(UserWarning):
    """
    Raised when a new operator has the same name as an already existing operator
    of that type, and the old is about to be overwritten.
    """
    pass
