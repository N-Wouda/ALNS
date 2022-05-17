import numpy.random as rnd
from numpy.testing import (assert_, assert_almost_equal, assert_equal,
                           assert_raises)
from pytest import mark

from alns import ALNS, State
from alns.criteria import HillClimbing, SimulatedAnnealing
from alns.stopping_criteria import MaxIterations, MaxRuntime
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
    result = alns.iterate(One(), weights, HillClimbing(), MaxIterations(1))
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
        alns.iterate(One(), weights, HillClimbing(), MaxIterations(1))


def test_raises_missing_repair_operator():
    """
    Tests if the algorithm raises when no repair operators have been set.
    """
    alns = get_alns_instance(destroy_operators=[lambda state, rnd: None])

    # Pretend we have a destroy operator for the weight scheme, so that
    # does not raise an error.
    weights = SimpleWeights([1, 1, 1, 1], 1, 1, 0.95)

    with assert_raises(ValueError):
        alns.iterate(One(), weights, HillClimbing(), MaxIterations(1))


def test_zero_max_iterations():
    """
    Test that the algorithm return the initial solution when the
    stopping criterion is zero max iterations.
    """
    alns = get_alns_instance([lambda state, rnd: None],
                             [lambda state, rnd: None])

    initial_solution = One()
    weights = SimpleWeights([1, 1, 1, 1], 1, 1, .5)

    result = alns.iterate(initial_solution, weights, HillClimbing(), MaxIterations(0))

    assert_(result.best_state is initial_solution)


def test_zero_max_runtime():
    """
    Test that the algorithm return the initial solution when the
    stopping criterion is zero max runtime.
    """
    alns = get_alns_instance([lambda state, rnd: None],
                             [lambda state, rnd: None])

    initial_solution = One()
    weights = SimpleWeights([1, 1, 1, 1], 1, 1, .5)

    result = alns.iterate(initial_solution, weights, HillClimbing(), MaxRuntime(0))

    assert_(result.best_state is initial_solution)


def test_iterate_kwargs_are_correctly_passed_to_operators():

    def test_operator(state, rnd, item):
        assert_(item is orig_item)
        return state

    alns = get_alns_instance([lambda state, rnd, item: state], [test_operator])

    init_sol = One()
    weights = SimpleWeights([1, 1, 1, 1], 1, 1, .5)
    orig_item = object()

    alns.iterate(init_sol, weights, HillClimbing(), MaxIterations(10), item=orig_item)


def test_bugfix_pass_kwargs_to_on_best():
    """
    Exercises a bug where the on_best callback did not receive the kwargs passed
    to iterate().
    """
    def test_operator(state, rnd, item):
        assert_(item is orig_item)
        return Zero()  # better, so on_best is triggered

    alns = get_alns_instance([lambda state, rnd, item: state], [test_operator])
    alns.on_best(lambda state, rnd, item: state)

    init_sol = One()
    weights = SimpleWeights([1, 1, 1, 1], 1, 1, .5)
    orig_item = object()

    alns.iterate(init_sol, weights, HillClimbing(), MaxIterations(10), item=orig_item)


# EXAMPLES ---------------------------------------------------------------------


def test_trivial_example():
    """
    This tests the ALNS algorithm on a trivial example, where the initial
    solution is one, and any other operator returns zero.
    """
    alns = get_alns_instance([lambda state, rnd: Zero()],
                             [lambda state, rnd: Zero()])

    weights = SimpleWeights([1, 1, 1, 1], 1, 1, .5)
    result = alns.iterate(One(), weights, HillClimbing(), MaxIterations(100))

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

    result = alns.iterate(One(), weights, sa, MaxIterations(100))
    assert_almost_equal(result.best_state.objective(), desired, decimal=5)


@mark.parametrize("max_iterations", [1, 10, 100])
def test_nonnegative_max_iterations(max_iterations):
    """
    Test that the result statistics have size equal to max iterations (+1).
    """
    alns = get_alns_instance([lambda state, rnd: Zero()],
                             [lambda state, rnd: Zero()])

    initial_solution = One()
    weights = SimpleWeights([1, 1, 1, 1], 1, 1, .5)

    result = alns.iterate(initial_solution, weights, HillClimbing(), MaxIterations(max_iterations))

    assert_equal(len(result.statistics.objectives), max_iterations + 1)
    assert_equal(len(result.statistics.runtimes), max_iterations)


@mark.parametrize("max_runtime", [0.1, 0.5, 1])
def test_nonnegative_max_runtime(max_runtime):
    """
    Test that the result runtime statistics correspond to the stopping criterion.
    """
    alns = get_alns_instance([lambda state, rnd: Zero()],
                             [lambda state, rnd: Zero()])

    initial_solution = One()
    weights = SimpleWeights([1, 1, 1, 1], 1, 1, .5)

    result = alns.iterate(initial_solution, weights, HillClimbing(), MaxRuntime(max_runtime))

    assert_almost_equal(sum(result.statistics.runtimes), max_runtime, decimal=3)


# TODO test more complicated examples?
