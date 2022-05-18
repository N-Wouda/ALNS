import time

import pytest
from numpy.random import RandomState
from numpy.testing import assert_, assert_raises

from alns.stop import MaxRuntime
from alns.tests.states import Zero


def sleep(duration, get_now=time.perf_counter):
    """
    Custom sleep function. Built-in time.sleep function is not precise
    and has different performances depending on the OS, see
    https://stackoverflow.com/questions/1133857/how-accurate-is-pythons-time-sleep
    """
    now = get_now()
    end = now + duration
    while now < end:
        now = get_now()


@pytest.mark.parametrize("max_runtime", [-0.001, -1, -10.1])
def test_raise_negative_parameters(max_runtime: float):
    """
    Maximum runtime may not be negative.
    """
    with assert_raises(ValueError):
        MaxRuntime(max_runtime)


@pytest.mark.parametrize("max_runtime", [0.001, 1, 10.1])
def test_valid_parameters(max_runtime: float):
    """
    Does not raise for non-negative parameters.
    """
    MaxRuntime(max_runtime)


@pytest.mark.parametrize("max_runtime", [0.01, 0.1, 1])
def test_max_runtime(max_runtime):
    """
    Test if the max time parameter is correctly set.
    """
    stop = MaxRuntime(max_runtime)
    assert_(stop.max_runtime, max_runtime)


@pytest.mark.parametrize("max_runtime", [0.01, 0.05, 0.10])
def test_before_max_runtime(max_runtime):
    stop = MaxRuntime(max_runtime)
    rnd = RandomState()
    for _ in range(100):
        assert_(not stop(rnd, Zero(), Zero()))


@pytest.mark.parametrize("max_runtime", [0.01, 0.05, 0.10])
def test_after_max_runtime(max_runtime):
    stop = MaxRuntime(max_runtime)
    rnd = RandomState()
    stop(rnd, Zero(), Zero())  # Trigger the first time measurement
    sleep(max_runtime)

    for _ in range(100):
        assert_(stop(rnd, Zero(), Zero()))
