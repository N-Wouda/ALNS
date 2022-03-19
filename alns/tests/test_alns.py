import numpy.random as rnd
from numpy.testing import (assert_, assert_almost_equal, assert_equal,
                           assert_raises)
from pytest import mark

from alns import ALNS, State
from alns.criteria import HillClimbing, SimulatedAnnealing
from alns.weight_schemes import SimpleWeights
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

    weights = SimpleWeights([1, 1, 1, 1], 1, 1, .5)
    result = alns.iterate(One(), weights, HillClimbing(), 1)
    assert_equal(result.best_state.objective(), 10)


# OPERATORS --------------------------------------------------------------------


def test_add_destroy_operator():
    """
    Tests if adding a destroy operator correctly updates the number of
    operators available on the ALNS instance.
    """
    alns = ALNS()

    for count in [1, 2]:
        alns.add_destroy_operator(lambda state, rnd: state, name=str(count))
        assert_equal(len(alns.destroy_operators), count)


def test_add_destroy_operator_name():
    """
    Tests if adding a destroy operator without an explicit name correctly
    takes the function name instead.
    """

    def destroy_operator():  # placeholder
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
        alns.add_repair_operator(lambda state, rnd: state, name=str(count))
        assert_equal(len(alns.repair_operators), count)


def test_add_repair_operator_name():
    """
    Tests if adding a repair operator without an explicit name correctly
    takes the function name instead.
    """

    def repair_operator():  # placeholder
        pass

    alns = ALNS()

    alns.add_repair_operator(repair_operator)

    name, operator = alns.repair_operators[0]

    assert_equal(name, "repair_operator")
    assert_(operator is repair_operator)


# PARAMETERS -------------------------------------------------------------------


def test_raises_missing_destroy_operator():
    """
    Tests if the algorithm raises when no destroy operators have been set.
    """
    alns = get_alns_instance(repair_operators=[lambda state, rnd: None])

    # Pretend we have a destroy operator for the weight scheme, so that
    # does not raise an error.
    weights = SimpleWeights([1, 1, 1, 1], 1, 1, 0.95)

    with assert_raises(ValueError):
        alns.iterate(One(), weights, HillClimbing())


def test_raises_missing_repair_operator():
    """
    Tests if the algorithm raises when no repair operators have been set.
    """
    alns = get_alns_instance(destroy_operators=[lambda state, rnd: None])

    # Pretend we have a destroy operator for the weight scheme, so that
    # does not raise an error.
    weights = SimpleWeights([1, 1, 1, 1], 1, 1, 0.95)

    with assert_raises(ValueError):
        alns.iterate(One(), weights, HillClimbing())


def test_raises_negative_iterations():
    """
    The number of iterations should be non-negative, as zero is allowed.
    """
    alns = get_alns_instance([lambda state, rnd: None],
                             [lambda state, rnd: None])

    initial_solution = One()
    weights = SimpleWeights([1, 1, 1, 1], 1, 1, .5)

    # A negative iteration count is not understood, for obvious reasons.
    with assert_raises(ValueError):
        alns.iterate(initial_solution, weights, HillClimbing(), -1)

    # But zero should just return the initial solution.
    result = alns.iterate(initial_solution, weights, HillClimbing(), 0)

    assert_(result.best_state is initial_solution)


def test_iterate_kwargs_are_correctly_passed_to_operators():

    def test_operator(state, rnd, item):
        assert_(item is orig_item)
        return state

    alns = get_alns_instance([lambda state, rnd, item: state], [test_operator])

    init_sol = One()
    weights = SimpleWeights([1, 1, 1, 1], 1, 1, .5)
    orig_item = object()

    alns.iterate(init_sol, weights, HillClimbing(), 10, item=orig_item)


# EXAMPLES ---------------------------------------------------------------------


def test_trivial_example():
    """
    This tests the ALNS algorithm on a trivial example, where the initial
    solution is one, and any other operator returns zero.
    """
    alns = get_alns_instance([lambda state, rnd: Zero()],
                             [lambda state, rnd: Zero()])

    weights = SimpleWeights([1, 1, 1, 1], 1, 1, .5)
    result = alns.iterate(One(), weights, HillClimbing(), 100)

    assert_equal(result.best_state.objective(), 0)


@mark.parametrize("seed,desired", [(0, 0.01171), (1, 0.00011), (2, 0.01025)])
def test_fixed_seed_outcomes(seed: int, desired: float):
    """
    Tests if fixing a seed results in deterministic outcomes even when using a
    'random' acceptance criterion (here SA).
    """
    alns = get_alns_instance(
        [lambda state, rnd: ValueState(rnd.random_sample())],
        [lambda state, rnd: None],
        seed)

    weights = SimpleWeights([1, 1, 1, 1], 1, 1, .5)
    sa = SimulatedAnnealing(1, .25, 1 / 100)

    result = alns.iterate(One(), weights, sa, 100)
    assert_almost_equal(result.best_state.objective(), desired, decimal=5)

# TODO test more complicated examples?
