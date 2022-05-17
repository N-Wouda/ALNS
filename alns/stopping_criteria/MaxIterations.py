from alns.stopping_criteria.StoppingCriterion import StoppingCriterion


class MaxIterations(StoppingCriterion):
    def __init__(self, max_iterations: int) -> None:
        """
        Criterion that stops after a maximum number of iterations.
        """
        if max_iterations < 0:
            raise ValueError("Max iterations must be non-negative.")

        self._max_iterations = max_iterations
        self._current_iteration = 0

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def current_iteration(self) -> int:
        return self._current_iteration

    def __call__(self) -> bool:
        self._current_iteration += 1

        return self.current_iteration > self.max_iterations
