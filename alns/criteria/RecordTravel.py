from .AcceptanceCriterion import AcceptanceCriterion


class RecordTravel(AcceptanceCriterion):

    def accept(self, best, current, candidate):
        return NotImplemented
