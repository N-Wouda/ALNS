class VarObj:
    """Test solution state object with variable objective."""

    def __init__(self, obj: float):
        self.obj = obj

    def objective(self) -> float:
        return self.obj


class ContextualVarObj:
    """Test solution state object with variable objective and context."""

    def __init__(self, obj: float, context: list):
        self.obj = obj
        self.context = context

    def objective(self) -> float:
        return self.obj

    def get_context(self) -> list:
        return self.context


Sentinel = lambda: VarObj(0)  # noqa: E731
Zero = lambda: VarObj(0)  # noqa: E731
One = lambda: VarObj(1)  # noqa: E731
Two = lambda: VarObj(2)  # noqa: E731

ZeroWithZeroContext = lambda: ContextualVarObj(0, [0, 0, 0])  # noqa: E731
ZeroWithOneContext = lambda: ContextualVarObj(1, [1, 1, 1])  # noqa: E731
