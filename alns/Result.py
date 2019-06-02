class Result:

    def __init__(self, best, last):
        self._best = best
        self._last = last

    @property
    def best_state(self):
        return self._best

    @property
    def last_state(self):
        return self._last
