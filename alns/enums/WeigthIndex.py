from enum import IntEnum, unique


@unique
class WeightIndex(IntEnum):
    """
    Names the various indices in the weights list. See eq. (1), p. 12 in
    Pisinger and Røpke (2010).
    """
    IS_BEST = 0
    IS_BETTER = 1
    IS_ACCEPTED = 2
    IS_REJECTED = 3
