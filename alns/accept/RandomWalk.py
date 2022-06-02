from alns.accept.AcceptanceCriterion import AcceptanceCriterion


class RandomWalk(AcceptanceCriterion):
    """
    The random walk criterion always accepts the candidate solution.
    """

    def __call__(self, rnd, best, current, candidate):
        return True
