import numpy as np
import numpy.random as rnd
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)
from pytest import mark

from alns.accept import SimulatedAnnealing
from alns.tests.states import One, Zero


@mark.parametrize(
    "start,end,step",
    [
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
    ],
)
def test_raises_negative_parameters(start: float, end: float, step: float):
    """
    Simulated annealing does not work with negative parameters, so those should
    not be accepted.
    """
    with assert_raises(ValueError):
        SimulatedAnnealing(start, end, step)


def test_raises_explosive_step():
    """
    For exponential updating, the step parameter must not be bigger than one,
    as that would result in an explosive threshold.
    """
    with assert_raises(ValueError):
        SimulatedAnnealing(2, 1, 2, "exponential")

    SimulatedAnnealing(2, 1, 1, "exponential")  # boundary should be fine


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

    SimulatedAnnealing(1, 1, 1)  # should not raise for equality


def test_does_not_raise():
    """
    These sets of parameters should work correctly.
    """
    SimulatedAnnealing(10, 5, 1, "exponential")
    SimulatedAnnealing(10, 5, 2, "linear")


@mark.parametrize("step", range(10))
def test_step(step: int):
    """
    Tests if the step parameter is correctly set.
    """
    # Linear because exponential does not take explosive step parameters.
    assert_equal(SimulatedAnnealing(1, 1, step, "linear").step, step)


@mark.parametrize("start", range(1, 10))
def test_start_temperature(start: int):
    """
    Tests if the start_temperature parameter is correctly set.
    """
    assert_equal(SimulatedAnnealing(start, 1, 1).start_temperature, start)


@mark.parametrize("end", range(1, 10))
def test_end_temperature(end: float):
    """
    Tests if the end_temperature parameter is correctly set.
    """
    assert_equal(SimulatedAnnealing(10, end, 1).end_temperature, end)


def test_accepts_better():
    for _ in range(1, 100):
        simulated_annealing = SimulatedAnnealing(2, 1, 1)
        assert_(simulated_annealing(rnd.RandomState(), One(), Zero(), Zero()))


def test_accepts_equal():
    simulated_annealing = SimulatedAnnealing(2, 1, 1)

    for _ in range(100):
        # This results in an acceptance probability of exp{0}, that is, one.
        # Thus, the candidate state should always be accepted.
        assert_(simulated_annealing(rnd.RandomState(), One(), One(), One()))


def test_linear_random_solutions():
    """
    Checks if the linear ``accept`` method correctly decides in two known cases
    for a fixed seed.
    """
    simulated_annealing = SimulatedAnnealing(2, 1, 1, "linear")

    state = rnd.RandomState(0)

    # Using the above seed, the first two random numbers are 0.55 and .72,
    # respectively. The acceptance probability is 0.61 first, so the first
    # should be accepted (0.61 > 0.55). Thereafter, it drops to 0.37, so the
    # second should not (0.37 < 0.72).
    assert_(simulated_annealing(state, Zero(), Zero(), One()))
    assert_(not simulated_annealing(state, Zero(), Zero(), One()))


def test_exponential_random_solutions():
    """
    Checks if the exponential ``accept`` method correctly decides in two known
    cases for a fixed seed. This is the exponential equivalent to the linear
    random solutions test above.
    """
    simulated_annealing = SimulatedAnnealing(2, 1, 0.5, "exponential")

    state = rnd.RandomState(0)

    assert_(simulated_annealing(state, Zero(), Zero(), One()))
    assert_(not simulated_annealing(state, Zero(), Zero(), One()))


def test_accepts_generator_and_random_state():
    """
    Tests if SimulatedAnnealing works with both Generator and RandomState
    randomness classes.

    See also https://numpy.org/doc/1.18/reference/random/index.html#quick-start
    """

    class Old:  # old RandomState interface mock
        def random_sample(self):  # pylint: disable=no-self-use
            return 0.5

    simulated_annealing = SimulatedAnnealing(2, 1, 1)
    assert_(simulated_annealing(Old(), One(), One(), Zero()))

    class New:  # new Generator interface mock
        def random(self):  # pylint: disable=no-self-use
            return 0.5

    simulated_annealing = SimulatedAnnealing(2, 1, 1)
    assert_(simulated_annealing(New(), One(), One(), Zero()))


@mark.parametrize(
    "worse,accept_prob,iters",
    [
        (1, 0, 10),  # zero accept prob
        (1, 1.2, 10),  # prob outside unit interval
        (1, 1, 10),  # unit accept prob
        (-1, 0.5, 10),  # negative worse
        (0, -1, 10),  # negative prob
        (1.5, 0.5, 10),  # worse outside unit interval
        (1, 0.9, -10),
    ],
)  # negative number of iterations
def test_autofit_raises_for_invalid_inputs(
    worse: float, accept_prob: float, iters: int
):
    with assert_raises(ValueError):
        SimulatedAnnealing.autofit(1.0, worse, accept_prob, iters)


@mark.parametrize(
    "init_obj,worse,accept_prob,iters",
    [(1_000, 1, 0.9, 1), (1_000, 0.5, 0.05, 1)],
)
def test_autofit_on_several_examples(
    init_obj: float, worse: float, accept_prob: float, iters: int
):
    # We have:
    # prob = exp{-(f^c - f^i) / T},
    # where T is start temp, f^i is init sol objective, and f^c is the candidate
    # solution objective. We also have that f^c is at worst (1 + worse) f^i.
    # Substituting and solving for T, we then find:
    # T = -worse * f^i / ln(p).
    sa_start = -worse * init_obj / np.log(accept_prob)
    sa_end = 1

    # We have end = r ** iters * start, so r = (end / start) ** (1 / iters).
    sa_step = (sa_end / sa_start) ** (1 / iters)

    sa = SimulatedAnnealing.autofit(init_obj, worse, accept_prob, iters)

    assert_almost_equal(sa.start_temperature, sa_start)
    assert_almost_equal(sa.end_temperature, sa_end)
    assert_almost_equal(sa.step, sa_step)
    assert_equal(sa.method, "exponential")
