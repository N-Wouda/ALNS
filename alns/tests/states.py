class VarObj:
    """Test solution state object with variable objective."""

    def __init__(self, obj: float):
        self.obj = obj

    def objective(self) -> float:
        return self.obj


Sentinel = lambda: VarObj(0)  # noqa: E731
Zero = lambda: VarObj(0)  # noqa: E731
One = lambda: VarObj(1)  # noqa: E731
Two = lambda: VarObj(2)  # noqa: E731
