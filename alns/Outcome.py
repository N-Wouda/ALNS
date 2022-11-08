from enum import IntEnum


class Outcome(IntEnum):
    """
    Enum of evaluation outcomes. A candidate solution can be a new global best,
    a better solution than the current incumbent, just accepted (but not
    improving anything), or rejected.
    """

    BEST = 0  #: Candidate solution is a new global best.
    BETTER = 1  #: Candidate solution is better than the current incumbent.
    ACCEPT = 2  #: Candidate solution is accepted.
    REJECT = 3  #: Candidate solution is rejected.
