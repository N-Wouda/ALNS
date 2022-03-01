from alns.criteria.AcceptanceCriterion import AcceptanceCriterion


class HillClimbing(AcceptanceCriterion):
    """
    Hill climbing only accepts progressively better solutions, discarding those
    that result in a worse objective value.
    """

    def __call__(self, rnd, best, current, candidate):
        return candidate.objective() <= current.objective()
