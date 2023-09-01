import numpy.random as rnd
from numpy.testing import assert_, assert_allclose, assert_equal, assert_raises
from pytest import mark

from alns.accept import RecordToRecordTravel
from alns.tests.states import One, Zero


@mark.parametrize(
    "start, end, step, method",
    [
        (-1, 1, 1, "linear"),  # start threshold cannot be negative
        (1, -1, 1, "linear"),  # nor can the end threshold
        (1, 1, -1, "linear"),  # nor the updating step
        (0, 1, 1, "linear"),  # start threshold cannot be < end threshold
        (2, 1, 2, "exponential"),  # step cannot be > 1 with exponential
        (1, 1, 1, "non-linear"),  # method must be 'linear' or 'exponential'
    ],
)
def test_raises_invalid_parameters(start, end, step, method):
    with assert_raises(ValueError):
        RecordToRecordTravel(start, end, step, method)


@mark.parametrize(
    "start, end, step, method",
    [
        (0, 0, 1, "linear"),  # boundary start and end threshold
        (0.5, 0.5, 1, "linear"),  # non-boundary equal start and end
        (10, 0, 0, "linear"),  # boundary step
        (1, 0.5, 1, "exponential"),  # boundary step exponential
        (1000, 0.5, 0.1, "linear"),  # regular parameters
    ],
)
def test_no_raise_valid_parameters(start, end, step, method):
    RecordToRecordTravel(start, end, step, method)


@mark.parametrize(
    "start, end, step, method",
    [
        (1, 0, 1, "linear"),
        (0.5, 0.5, 1, "linear"),
        (10, 5, 0.001, "linear"),
        (1, 0.5, 1, "exponential"),
    ],
)
def test_properties(start, end, step, method):
    """
    Test if the properties are correctly set.
    """
    rrt = RecordToRecordTravel(start, end, step, method)
    assert_equal(rrt.start_threshold, start)
    assert_equal(rrt.end_threshold, end)
    assert_equal(rrt.step, step)
    assert_equal(rrt.method, method)


def test_accepts_better():
    record_travel = RecordToRecordTravel(1, 0, 0.1)
    assert_(record_travel(rnd.default_rng(), One(), Zero(), Zero()))


def test_rejects_worse():
    record_travel = RecordToRecordTravel(0.5, 0.2, 0.1)

    # This results in a relative worsening of plus one, which is bigger than
    # the threshold (0.5).
    assert_(not record_travel(rnd.default_rng(), Zero(), Zero(), One()))


def test_accepts_equal():
    record_travel = RecordToRecordTravel(0, 0, 0.1)

    # Even a the strictest threshold, this should be accepted since the
    # relative improvement is zero (they are equal).
    assert_(record_travel(rnd.default_rng(), Zero(), Zero(), Zero()))


def test_linear_threshold_update():
    record_travel = RecordToRecordTravel(5, 0, 1)

    # For the first five, the threshold is, resp., 5, 4, 3, 2, 1. The relative
    # worsening is plus one, so this should be accepted (lower or equal to
    # threshold).
    for _ in range(5):
        assert_(record_travel(rnd.default_rng(), Zero(), Zero(), One()))

    # Threshold is now zero, so this should no longer be accepted.
    assert_(not record_travel(rnd.default_rng(), Zero(), Zero(), One()))


def test_exponential_threshold_update():
    record_travel = RecordToRecordTravel(5, 0, 0.1, "exponential")

    # The relative worsening is plus one, so this should be accepted initially,
    # as the threshold is 5, resp. 0.5. In the second case, 1 > 0.5, so the
    # second should be rejected.
    assert_(record_travel(rnd.default_rng(), Zero(), Zero(), One()))
    assert_(not record_travel(rnd.default_rng(), Zero(), Zero(), One()))


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
    "init_obj,start_gap,end_gap,n_iters,method,exp_start,exp_end,exp_step",
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
    exp_start: float,
    exp_end: float,
    exp_step: float,
):
    rrt = RecordToRecordTravel.autofit(
        init_obj, start_gap, end_gap, n_iters, method
    )

    assert_allclose(rrt.start_threshold, exp_start)
    assert_allclose(rrt.end_threshold, exp_end)
    assert_allclose(rrt.step, exp_step, rtol=1e-3)
    assert_equal(rrt.method, method)
