from enum import Enum


class WeightIndex(Enum):
    """
    Names the various indices in the weights list.
    """

    IS_BEST = 0
    IS_BETTER = 1
    IS_ACCEPTED = 2
    IS_REJECTED = 3
