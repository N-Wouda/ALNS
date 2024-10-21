class AlwaysAccept:
    """
    This criterion always accepts the candidate solution.
    """

    def __call__(self, rng, best, current, candidate):
        return True
