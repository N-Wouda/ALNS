from enum import IntEnum, unique


@unique
class CallbackFlag(IntEnum):
    """
    Callback flags for the mix-in.
    """
    ON_BEST = 0
