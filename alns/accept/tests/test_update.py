from alns.accept.update import update
from numpy.testing import assert_raises, assert_equal


def test_raises_unknown_method():
    with assert_raises(ValueError):
        update(1, 0.5, "unknown_method")

    update(1, 0.5, "linear")  # this should work


def test_accepts_any_case_method():
    """
    ``update`` should be indifferent about the passed-in method case.
    """
    assert_equal(update(1, 0.5, "linear"), update(1, 0.5, "LINEAR"))
    assert_equal(update(1, 0.5, "exponential"), update(1, 0.5, "EXPONENTIAL"))


def test_linear():
    """
    Tests if linear updating works as designed, as ``current - step``.
    """
    assert_equal(update(1, 0.5, "linear"), 0.5)
    assert_equal(update(2, 1, "linear"), 1)
    assert_equal(update(2, 0, "linear"), 2)


def test_exponential():
    """
    Tests if exponential updating works as designed, as ``current * step``.
    """
    assert_equal(update(1, 0.5, "exponential"), 0.5)
    assert_equal(update(2, 1, "exponential"), 2)
    assert_equal(update(2, 0, "exponential"), 0)
