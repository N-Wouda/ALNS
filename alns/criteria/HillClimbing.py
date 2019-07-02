from .AcceptanceCriterion import AcceptanceCriterion


class HillClimbing(AcceptanceCriterion):
    """
    Hill climbing only accepts progressively better solutions, discarding those
    that result in a worse objective value.
    """

    def accept(self, rnd, best, current, candidate):
        return candidate.objective() <= current.objective()
