import pytest
from numpy.testing import assert_equal, assert_raises

from alns.accept.update import update


def test_raises_unknown_method():
    with assert_raises(ValueError):
        update(1, 0.5, "unknown_method")

    update(1, 0.5, "linear")  # this should work


@pytest.mark.parametrize(
    "lc,uc",
    [
        ("linear", "LINEAR"),
        ("linear", "Linear"),
        ("exponential", "EXPONENTIAL"),
    ],
)
def test_accepts_any_case_method(lc: str, uc: str):
    """
    ``update`` should be indifferent about the passed-in method case.
    """
    assert_equal(update(1, 0.5, lc), update(1, 0.5, uc))


@pytest.mark.parametrize(
    "curr,step,method,expected",
    [
        (1, 0.5, "linear", 0.5),
        (2, 1, "linear", 1),
        (2, 0, "linear", 2),
        (1, 0.5, "exponential", 0.5),
        (2, 1, "exponential", 2),
        (2, 0, "exponential", 0),
    ],
)
def test_update(curr: float, step: float, method: str, expected: float):
    """
    Tests if linear updating works as designed, as ``current - step``, and
    exponential updating as ``current * step``.
    """
    assert_equal(update(curr, step, method), expected)
