from alns.accept.AcceptanceCriterion import AcceptanceCriterion


class RandomWalk(AcceptanceCriterion):
    """
    RandomWalk always accepts the candidate solution.
    """

    def __call__(self, rnd, best, current, candidate):
        return True
