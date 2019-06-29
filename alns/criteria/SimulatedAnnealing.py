from .AcceptanceCriterion import AcceptanceCriterion


class SimulatedAnnealing(AcceptanceCriterion):

    def __init__(self):
        """
        TODO
        """
        pass

    def accept(self, best, current, candidate):
        return NotImplemented
