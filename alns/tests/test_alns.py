import numpy.random as rnd
from numpy.testing import (assert_, assert_almost_equal, assert_equal,
                           assert_raises, assert_no_warnings, assert_warns)

from alns import ALNS, State
from alns.criteria import HillClimbing, SimulatedAnnealing
from alns.exceptions_warnings import OverwriteWarning
from .states import One, Zero


# HELPERS ----------------------------------------------------------------------


def get_alns_instance(repair_operators=None, destroy_operators=None, seed=None):
    """
    Test helper method.
    """
    alns = ALNS(rnd.RandomState(seed))

    if repair_operators is not None:
        for idx, repair_operator in enumerate(repair_operators):
            alns.add_repair_operator(repair_operator, name=str(idx))

    if destroy_operators is not None:
        for idx, destroy_operator in enumerate(destroy_operators):
            alns.add_destroy_operator(destroy_operator, name=str(idx))

    return alns


class ValueState(State):
    """
    Helper state for testing random values.
    """

    def __init__(self, value):
        self._value = value

    def objective(self):
        return self._value


# CALLBACKS --------------------------------------------------------------------

def test_on_best_is_called():
    """
    Tests if the callback is invoked when a new global best is found.
    """
    alns = get_alns_instance([lambda state, rnd: Zero()],
                             [lambda state, rnd: Zero()])

    # Called when a new global best is found. In this case, that happens once:
    # in the only iteration below. It returns a state with value 10, which
    # should then also be returned by the entire algorithm.
    alns.on_best(lambda *args: ValueState(10))

    result = alns.iterate(One(), [1, 1, 1, 1], .5, HillClimbing(), 1)
    assert_equal(result.best_state.objective(), 10)

# OPERATORS --------------------------------------------------------------------


def test_add_destroy_operator():
    """
    Tests if adding a destroy operator correctly updates the number of
    operators available on the ALNS instance.
    """
    alns = ALNS()

    for count in [1, 2]:
        alns.add_destroy_operator(lambda state, rnd: None, name=str(count))
        assert_equal(len(alns.destroy_operators), count)


def test_add_destroy_operator_name():
    """
    Tests if adding a destroy operator without an explicit name correctly
    takes the function name instead.
    """
    def destroy_operator():                 # placeholder
        pass

    alns = ALNS()

    alns.add_destroy_operator(destroy_operator)

    name, operator = alns.destroy_operators[0]

    assert_equal(name, "destroy_operator")
    assert_(operator is destroy_operator)


def test_add_repair_operator():
    """
    Tests if adding a repair operator correctly updates the number of
    operators available on the ALNS instance.
    """
    alns = ALNS()

    for count in [1, 2]:
        alns.add_repair_operator(lambda state, rnd: None, name=str(count))
        assert_equal(len(alns.repair_operators), count)


def test_add_repair_operator_name():
    """
    Tests if adding a repair operator without an explicit name correctly
    takes the function name instead.
    """
    def repair_operator():                  # placeholder
        pass

    alns = ALNS()

    alns.add_repair_operator(repair_operator)

    name, operator = alns.repair_operators[0]

    assert_equal(name, "repair_operator")
    assert_(operator is repair_operator)


def test_add_operator_same_name_warns_per_type():
    """
    Adding an operator with the same name as an already added operator (of the
    same type) should warn that the previous operator will be overwritten.
    """
    alns = ALNS()

    with assert_no_warnings():
        # The same name is allowed for different types of operators.
        alns.add_destroy_operator(lambda state, rnd: None, "test")
        alns.add_repair_operator(lambda state, rnd: None, "test")

    with assert_warns(OverwriteWarning):
        # Already exists as a destroy operator.
        alns.add_destroy_operator(lambda state, rnd: None, "test")

    with assert_warns(OverwriteWarning):
        # Already exists as a repair operator.
        alns.add_repair_operator(lambda state, rnd: None, "test")


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


def test_raises_negative_iterations():
    """
    The number of iterations should be non-negative, as zero is allowed.
    """
    alns = get_alns_instance([lambda state, rnd: None],
                             [lambda state, rnd: None])

    initial_solution = One()

    # A negative iteration count is not understood, for obvious reasons.
    with assert_raises(ValueError):
        alns.iterate(initial_solution, [1, 1, 1, 1], .5, HillClimbing(), -1)

    # But zero should just return the initial solution.
    result = alns.iterate(initial_solution, [1, 1, 1, 1], .5, HillClimbing(), 0)

    assert_(result.best_state is initial_solution)


def test_does_not_raise():
    """
    This set of parameters, on the other hand, should work correctly.
    """
    alns = get_alns_instance([lambda state, rnd: One()],
                             [lambda state, rnd: One()])

    alns.iterate(Zero(), [1, 1, 1, 1], .5, HillClimbing(), 100)

    # 0 and 1 are both acceptable decay parameters (since v1.2.0).
    alns.iterate(Zero(), [1, 1, 1, 1], 0., HillClimbing(), 100)
    alns.iterate(Zero(), [1, 1, 1, 1], 1., HillClimbing(), 100)


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
    outcomes = [0.01171, 0.00011, 0.01025]

    for seed, desired in enumerate(outcomes):                   # idx is seed
        alns = get_alns_instance(
            [lambda state, rnd: ValueState(rnd.random_sample())],
            [lambda state, rnd: None],
            seed)

        simulated_annealing = SimulatedAnnealing(1, .25, 1 / 100)

        result = alns.iterate(One(), [1, 1, 1, 1], .5, simulated_annealing, 100)

        assert_almost_equal(result.best_state.objective(), desired, decimal=5)

# TODO test more complicated examples?
