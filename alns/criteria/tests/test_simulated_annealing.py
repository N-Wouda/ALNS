import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises

from alns.criteria import SimulatedAnnealing
from alns.tests.states import One, Zero


def test_raises_negative_parameters():
    """
    Simulated annealing does not work with negative parameters, so those should
    not be accepted.
    """
    with assert_raises(ValueError):         # start temperature cannot be
        SimulatedAnnealing(-1, 1, 1)        # negative

    with assert_raises(ValueError):         # nor can the end temperature
        SimulatedAnnealing(1, -1, 1)

    with assert_raises(ValueError):         # nor the updating step
        SimulatedAnnealing(1, 1, -1)


def test_raises_explosive_step():
    """
    For exponential updating, the step parameter must not be bigger than one,
    as that would result in an explosive threshold.
    """
    with assert_raises(ValueError):
        SimulatedAnnealing(2, 1, 2, "exponential")

    SimulatedAnnealing(2, 1, 1, "exponential")    # boundary should be fine


def test_temperature_boundary():
    """
    The boundary case for the end temperature parameter is at zero, which must
    *not* be accepted, as it would result in a division-by-zero.
    """
    with assert_raises(ValueError):
        SimulatedAnnealing(1, 0, 1)


def test_raises_start_smaller_than_end():
    """
    The initial temperature at the start should be bigger (or equal) to the end
    temperature.
    """
    with assert_raises(ValueError):
        SimulatedAnnealing(0.5, 1, 1)

    SimulatedAnnealing(1, 1, 1)           # should not raise for equality


def test_does_not_raise():
    """
    This set of parameters should work correctly.
    """
    SimulatedAnnealing(10, 5, 2)


def test_step():
    """
    Tests if the step parameter is correctly set.
    """
    for step in range(100):
        assert_equal(SimulatedAnnealing(1, 1, step).step, step)


def test_start_temperature():
    """
    Tests if the start_temperature parameter is correctly set.
    """
    for start in range(1, 100):
        assert_equal(SimulatedAnnealing(start, 1, 1).start_temperature, start)


def test_end_temperature():
    """
    Tests if the end_temperature parameter is correctly set.
    """
    for end in range(1, 100):
        assert_equal(SimulatedAnnealing(100, end, 1).end_temperature, end)


def test_accepts_better():
    for _ in range(1, 100):
        simulated_annealing = SimulatedAnnealing(2, 1, 1)

        assert_(simulated_annealing.accept(rnd.RandomState(),
                                           One(),
                                           Zero(),
                                           Zero()))


def test_accepts_equal():
    simulated_annealing = SimulatedAnnealing(2, 1, 1)

    for _ in range(100):
        # This results in an acceptance probability of exp{0}, that is, one.
        # Thus, the candidate state should always be accepted.
        assert_(simulated_annealing.accept(rnd.RandomState(),
                                           One(),
                                           One(),
                                           One()))


def test_linear_random_solutions():
    """
    Checks if the linear ``accept`` method correctly decides in two known cases
    for a fixed seed.
    """
    simulated_annealing = SimulatedAnnealing(2, 1, 1)

    state = rnd.RandomState(0)

    # Using the above seed, the first two random numbers are 0.55 and .72,
    # respectively. The acceptance probability is 0.61 first, so the first
    # should be accepted (0.61 > 0.55). Thereafter, it drops to 0.37, so the
    # second should not (0.37 < 0.72).
    assert_(simulated_annealing.accept(state, Zero(), Zero(), One()))
    assert_(not simulated_annealing.accept(state, Zero(), Zero(), One()))


def test_exponential_random_solutions():
    """
    Checks if the exponential ``accept`` method correctly decides in two known
    cases for a fixed seed. This is the exponential equivalent to the linear
    random solutions test above.
    """
    simulated_annealing = SimulatedAnnealing(2, 1, 0.5, "exponential")

    state = rnd.RandomState(0)

    assert_(simulated_annealing.accept(state, Zero(), Zero(), One()))
    assert_(not simulated_annealing.accept(state, Zero(), Zero(), One()))

