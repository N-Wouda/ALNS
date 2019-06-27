from abc import ABC, abstractmethod


class AcceptanceCriterion(ABC):

    @abstractmethod
    def accept(self, best, current, candidate):
        return NotImplemented
