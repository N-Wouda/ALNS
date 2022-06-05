from numbers import Number

from alns import State


class VarObj(State):
    """Test solution state object with variable objective."""

    def __init__(self, obj: Number):
        self.obj = obj

    def objective(self) -> Number:
        return self.obj


Sentinel = lambda: VarObj(0)
Zero = lambda: VarObj(0)
One = lambda: VarObj(1)
Two = lambda: VarObj(2)
