import numpy.random as rnd
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

from alns import ALNS, State
from alns.criteria import HillClimbing, SimulatedAnnealing
from .states import One, Zero


# HELPERS ----------------------------------------------------------------------


def get_alns_instance(repair_operators=None, destroy_operators=None, seed=None):
    """
    Test helper method.
    """
    alns = ALNS(rnd.RandomState(seed))

    if repair_operators is not None:
        for repair_operator in repair_operators:
            alns.add_repair_operator(repair_operator)

    if destroy_operators is not None:
        for destroy_operator in destroy_operators:
            alns.add_destroy_operator(destroy_operator)

    return alns


class ValueState(State):
    """
    Helper state for testing random values.
    """

    def __init__(self, value):
        self._value = value

    def objective(self):
        return self._value


# OPERATORS --------------------------------------------------------------------


def test_add_destroy_operator():
    """
    Tests if adding a destroy operator correctly updates the number of
    operators available on the ALNS instance.
    """
    alns = ALNS()

    for count in [1, 2]:
        alns.add_destroy_operator(lambda state, rnd: None)
        assert_equal(len(alns.destroy_operators), count)


def test_add_repair_operator():
    """
    Tests if adding a repair operator correctly updates the number of
    operators available on the ALNS instance.
    """
    alns = ALNS()

    for count in [1, 2]:
        alns.add_repair_operator(lambda state, rnd: None)
        assert_equal(len(alns.repair_operators), count)


# PARAMETERS -------------------------------------------------------------------


def test_raises_missing_destroy_operator():
    """
    Tests if the algorithm raises when no destroy operators have been set.
    """
    alns = get_alns_instance([lambda state, rnd: None])

    with assert_raises(ValueError):
        alns.iterate(One(), [1, 1, 1, 1], 0.95, HillClimbing())


def test_raises_missing_repair_operator():
    """
    Tests if the algorithm raises when no repair operators have been set.
    """
    alns = get_alns_instance(destroy_operators=[lambda state, rnd: None])

    with assert_raises(ValueError):
        alns.iterate(One(), [1, 1, 1, 1], 0.95, HillClimbing())


def test_raises_negative_operator_decay():
    """
    Tests if the algorithm raises when a negative operator decay parameter is
    passed.
    """
    alns = get_alns_instance([lambda state, rnd: None],
                             [lambda state, rnd: None])

    with assert_raises(ValueError):
        alns.iterate(One(), [1, 1, 1, 1], -0.5, HillClimbing())


def test_raises_explosive_operator_decay():
    """
    Tests if the algorithm raises when an explosive operator decay parameter is
    passed.
    """
    alns = get_alns_instance([lambda state, rnd: None],
                             [lambda state, rnd: None])

    with assert_raises(ValueError):
        alns.iterate(One(), [1, 1, 1, 1], 1.2, HillClimbing())


def test_raises_boundary_operator_decay():
    """
    The boundary cases, zero and one, should both raise.
    """
    alns = get_alns_instance([lambda state, rnd: None],
                             [lambda state, rnd: None])

    with assert_raises(ValueError):
        alns.iterate(One(), [1, 1, 1, 1], 0, HillClimbing())

    with assert_raises(ValueError):
        alns.iterate(One(), [1, 1, 1, 1], 1, HillClimbing())


def test_raises_insufficient_weights():
    """
    We need (at least) four weights to be passed-in, one for each updating
    scenario.
    """
    alns = get_alns_instance([lambda state, rnd: None],
                             [lambda state, rnd: None])

    with assert_raises(ValueError):
        alns.iterate(One(), [1, 1, 1], .5, HillClimbing())


def test_raises_non_positive_weights():
    """
    The passed-in weights should all be strictly positive.
    """
    alns = get_alns_instance([lambda state, rnd: None],
                             [lambda state, rnd: None])

    with assert_raises(ValueError):
        alns.iterate(One(), [1, 1, 0, 1], .5, HillClimbing())

    with assert_raises(ValueError):
        alns.iterate(One(), [1, 1, -5, 1], .5, HillClimbing())


def test_raises_non_positive_iterations():
    """
    The number of iterations should be strictly positive.
    """
    alns = get_alns_instance([lambda state, rnd: None],
                             [lambda state, rnd: None])

    with assert_raises(ValueError):
        alns.iterate(One(), [1, 1, 1, 1], .5, HillClimbing(), 0)

    with assert_raises(ValueError):
        alns.iterate(One(), [1, 1, 1, 1], .5, HillClimbing(), -1)


def test_does_not_raise():
    """
    This set of parameters, on the other hand, should work correctly.
    """
    alns = get_alns_instance([lambda state, rnd: One()],
                             [lambda state, rnd: One()])

    alns.iterate(Zero(), [1, 1, 1, 1], .5, HillClimbing(), 100)


# EXAMPLES ---------------------------------------------------------------------


def test_trivial_example():
    """
    This tests the ALNS algorithm on a trivial example, where the initial
    solution is zero, and any other operator returns one.
    """
    alns = get_alns_instance([lambda state, rnd: Zero()],
                             [lambda state, rnd: Zero()])

    result = alns.iterate(One(), [1, 1, 1, 1], .5, HillClimbing(), 100)

    assert_equal(result.best_state.objective(), 0)


def test_fixed_seed_outcomes():
    """
    Tests if fixing a seed results in deterministic outcomes even when using a
    'random' acceptance criterion (here SA).
    """
    outcomes = [0.00469, 0.00011, 0.04312]

    for seed, desired in enumerate(outcomes):                   # idx is seed
        alns = get_alns_instance(
            [lambda state, rnd: ValueState(rnd.random_sample())],
            [lambda state, rnd: None],
            seed)

        simulated_annealing = SimulatedAnnealing(1, .25, 1 / 100)

        result = alns.iterate(One(), [1, 1, 1, 1], .5, simulated_annealing, 100)

        assert_almost_equal(result.best_state.objective(), desired, decimal=5)


# TODO test more complicated examples?
