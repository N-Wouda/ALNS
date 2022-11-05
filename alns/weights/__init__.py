import warnings

from alns.select import RouletteWheel as SimpleWeights
from alns.select import SegmentedRouletteWheel as SegmentedWeights

message = (
    "alns.weights has been deprecated in favour of alns.select."
    "See https://github.com/N-Wouda/ALNS/issues/74 for details."
)

warnings.warn(message, DeprecationWarning, stacklevel=2)
