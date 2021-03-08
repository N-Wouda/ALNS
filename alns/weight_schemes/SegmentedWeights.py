from .WeightScheme import WeightScheme


class SegmentedWeights(WeightScheme):
    pass


"""
    weights
            A list of four non-negative elements, representing the weight
            updates when the candidate solution results in a new global best
            (idx 0), is better than the current solution (idx 1), the solution
            is accepted (idx 2), or rejected (idx 3).
        op_decay : float
            The operator decay parameter, as a float in the unit interval,
            [0, 1] (inclusive).
        """
