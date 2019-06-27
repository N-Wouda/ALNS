from .AcceptanceCriterion import AcceptanceCriterion


class HillClimbing(AcceptanceCriterion):

    def accept(self, best, current, candidate):
        return NotImplemented
