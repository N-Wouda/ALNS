from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)
from pytest import mark

from alns.accept import RecordToRecordTravel
from alns.tests.states import One, Zero


def test_raises_negative_parameters():
    """
    Record-to-record travel does not work with negative parameters, so those
    should not be accepted.
    """
    with assert_raises(ValueError):  # start threshold cannot be negative
        RecordToRecordTravel(-1, 1, 1)

    with assert_raises(ValueError):  # nor can the end threshold
        RecordToRecordTravel(1, -1, 1)

    with assert_raises(ValueError):  # nor the updating step
        RecordToRecordTravel(1, 1, -1)


def test_raises_explosive_step():
    """
    For exponential updating, the step parameter must not be bigger than one,
    as that would result in an explosive threshold.
    """
    with assert_raises(ValueError):
        RecordToRecordTravel(2, 1, 2, "exponential")

    RecordToRecordTravel(2, 1, 1, "exponential")  # boundary should be fine


def test_raises_invalid_method():
    with assert_raises(ValueError):
        RecordToRecordTravel(1, 1, 1, "non-linear")


def test_threshold_boundary():
    """
    The boundary case for the end threshold parameter is at zero, which should
    be accepted.
    """
    RecordToRecordTravel(1, 0, 1)


def test_raises_start_smaller_than_end():
    """
    The initial threshold at the start should be bigger (or equal) to the end
    threshold.
    """
    with assert_raises(ValueError):
        RecordToRecordTravel(0, 1, 1)

    RecordToRecordTravel(1, 1, 1)  # should not raise for equality


def test_does_not_raise():
    """
    This set of parameters should work correctly.
    """
    RecordToRecordTravel(10, 5, 2)


def test_step():
    """
    Tests if the step parameter is correctly set.
    """
    for step in range(100):
        assert_equal(RecordToRecordTravel(1, 1, step).step, step)


def test_start_threshold():
    """
    Tests if the start_threshold parameter is correctly set.
    """
    for start in range(100):
        assert_equal(RecordToRecordTravel(start, 0, 1).start_threshold, start)


def test_end_threshold():
    """
    Tests if the end_threshold parameter is correctly set.
    """
    for end in range(100):
        assert_equal(RecordToRecordTravel(100, end, 1).end_threshold, end)


def test_accepts_better():
    record_travel = RecordToRecordTravel(1, 0, 0.1)
    assert_(record_travel(None, One(), None, Zero()))


def test_rejects_worse():
    record_travel = RecordToRecordTravel(0.5, 0.2, 0.1)

    # This results in a relative worsening of plus one, which is bigger than
    # the threshold (0.5).
    assert_(not record_travel(None, Zero(), None, One()))


def test_accepts_equal():
    record_travel = RecordToRecordTravel(0, 0, 0.1)

    # Even a the strictest threshold, this should be accepted since the
    # relative improvement is zero (they are equal).
    assert_(record_travel(None, Zero(), None, Zero()))


def test_linear_threshold_update():
    record_travel = RecordToRecordTravel(5, 0, 1)

    # For the first five, the threshold is, resp., 5, 4, 3, 2, 1. The relative
    # worsening is plus one, so this should be accepted (lower or equal to
    # threshold).
    for _ in range(5):
        assert_(record_travel(None, Zero(), None, One()))

    # Threshold is now zero, so this should no longer be accepted.
    assert_(not record_travel(None, Zero(), None, One()))


def test_exponential_threshold_update():
    record_travel = RecordToRecordTravel(5, 0, 0.1, "exponential")

    # The relative worsening is plus one, so this should be accepted initially,
    # as the threshold is 5, resp. 0.5. In the second case, 1 > 0.5, so the
    # second should be rejected.
    assert_(record_travel(None, Zero(), None, One()))
    assert_(not record_travel(None, Zero(), None, One()))


@mark.parametrize(
    "init_obj, start_gap, end_gap, n_iters, method",
    [
        (1, -1, 0, 1, "linear"),  # negative start
        (1, 0, -1, 1, "linear"),  # negative end
        (1, 0.5, 1, 1, "linear"),  # start < end
        (1, 0, 0, 0, "linear"),  # non-positive n_iters
        (1, 0, 0, 1, "nonlinear"),  # invalid method
    ],
)
def test_autofit_raises_for_invalid_inputs(
    init_obj: float,
    start_gap: float,
    end_gap: float,
    n_iters: int,
    method: str,
):
    with assert_raises(ValueError):
        RecordToRecordTravel.autofit(
            init_obj, start_gap, end_gap, n_iters, method
        )


@mark.parametrize(
    "init_obj, start_gap, end_gap, n_iters, method, rrt_start, rrt_end, rrt_step",
    [
        (1, 1, 0, 1, "linear", 1, 0, 1),
        (10, 0.1, 0, 10, "linear", 1, 0, 0.1),
        (100, 0.5, 0.5, 10, "linear", 50, 50, 0),
        (1, 1, 0.1, 1, "exponential", 1, 0.1, 0.1),
        (10, 0.1, 0.01, 10, "exponential", 1, 0.1, 0.79432),
        (100, 0.05, 0.05, 10, "exponential", 5, 5, 1),
    ],
)
def test_autofit_on_several_examples(
    init_obj: float,
    start_gap: float,
    end_gap: float,
    n_iters: int,
    method: str,
    rrt_start: float,
    rrt_end: float,
    rrt_step: float,
):
    rrt = RecordToRecordTravel.autofit(
        init_obj, start_gap, end_gap, n_iters, method
    )

    assert_almost_equal(rrt.start_threshold, rrt_start)
    assert_almost_equal(rrt.end_threshold, rrt_end)
    assert_almost_equal(rrt.step, rrt_step, decimal=3)
    assert_equal(rrt.method, method)
