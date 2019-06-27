from .AcceptanceCriterion import AcceptanceCriterion


class SimulatedAnnealing(AcceptanceCriterion):

    def accept(self, best, current, candidate):
        return NotImplemented
