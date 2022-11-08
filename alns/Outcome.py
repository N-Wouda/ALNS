from enum import IntEnum


class Outcome(IntEnum):
    """
    Evaluation outcomes. A candidate solution can be a new global best, a
    better solution than the current incumbent, just accepted (but not
    improving anything), or rejected.
    """

    BEST = 0
    BETTER = 1
    ACCEPT = 2
    REJECT = 3
