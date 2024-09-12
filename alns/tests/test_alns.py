import numpy.random as rnd
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)
from pytest import mark

from alns import ALNS
from alns.accept import HillClimbing, SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations, MaxRuntime

from .states import One, VarObj, Zero

# HELPERS ---------------------------------------------------------------------


def get_alns_instance(
    repair_operators=None, destroy_operators=None, seed=None
):
    """
    Test helper method.
    """
    alns = ALNS(rnd.default_rng(seed))

    if repair_operators is not None:
        for idx, repair_operator in enumerate(repair_operators):
            alns.add_repair_operator(repair_operator, name=str(idx))

    if destroy_operators is not None:
        for idx, destroy_operator in enumerate(destroy_operators):
            alns.add_destroy_operator(destroy_operator, name=str(idx))

    return alns


def get_repair_operators(n):
    """
    Get a list of n dummy repair operators.
    """
    repair_operators = []

    for idx in range(n):
        op = lambda: None  # noqa: E731
        op.__name__ = f"Repair operator {idx}"
        repair_operators.append(op)

    return repair_operators


def get_destroy_operators(n):
    """
    Get a list of n dummy destroy operators.
    """
    destroy_operators = []

    for idx in range(n):
        op = lambda: None  # noqa: E731
        op.__name__ = f"Destroy operator {idx}"
        destroy_operators.append(op)

    return destroy_operators


# CALLBACKS -------------------------------------------------------------------


def test_on_best_is_called():
    """
    Tests if the callback is invoked when a new global best is found.
    """
    alns = get_alns_instance(
        [lambda state, rng: Zero()], [lambda state, rng: Zero()]
    )

    # Called when a new global best is found. In this case, that happens once:
    # in the only iteration below. We change the objective, and test whether
    # that is indeed the solution that is returned
    def callback(state, rng):
        state.obj = 1

    alns.on_best(callback)

    select = RouletteWheel([1, 1, 1, 1], 0.5, 1, 1)
    result = alns.iterate(VarObj(10), select, HillClimbing(), MaxIterations(1))

    assert_equal(result.best_state.objective(), 1)


def test_other_callbacks_are_called():
    alns = get_alns_instance(
        [lambda state, rng: state],
        [lambda state, rng: VarObj(rng.random())],
        seed=1,
    )

    registry = dict(on_better=False, on_accept=False, on_reject=False)

    def mock_callback(state, rng, key):
        registry[key] = True

    alns.on_better(lambda state, rng: mock_callback(state, rng, "on_better"))
    alns.on_accept(lambda state, rng: mock_callback(state, rng, "on_accept"))
    alns.on_reject(lambda state, rng: mock_callback(state, rng, "on_reject"))

    select = RouletteWheel([1, 1, 1, 1], 0.5, 1, 1)
    accept = SimulatedAnnealing(1_000, 1, 0.05)
    alns.iterate(VarObj(10), select, accept, MaxIterations(1_000))

    assert_(registry["on_better"])
    assert_(registry["on_accept"])
    assert_(registry["on_reject"])


# OPERATORS -------------------------------------------------------------------


def test_add_destroy_operator():
    """
    Tests if adding a destroy operator correctly updates the number of
    operators available on the ALNS instance.
    """
    alns = ALNS()

    for count in [1, 2]:
        alns.add_destroy_operator(lambda state, rng: state, name=str(count))
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
        alns.add_repair_operator(lambda state, rng: state, name=str(count))
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


# PARAMETERS ------------------------------------------------------------------


def test_raises_missing_destroy_operator():
    """
    Tests if the algorithm raises when no destroy operators have been set.
    """
    alns = get_alns_instance(repair_operators=[lambda state, rng: None])

    # Pretend we have a destroy operator for the selection scheme, so that
    # does not raise an error.
    select = RouletteWheel([1, 1, 1, 1], 0.95, 1, 1)

    with assert_raises(ValueError):
        alns.iterate(One(), select, HillClimbing(), MaxIterations(1))


def test_raises_missing_repair_operator():
    """
    Tests if the algorithm raises when no repair operators have been set.
    """
    alns = get_alns_instance(destroy_operators=[lambda state, rng: None])

    # Pretend we have a destroy operator for the selection scheme, so that
    # does not raise an error.
    select = RouletteWheel([1, 1, 1, 1], 0.95, 1, 1)

    with assert_raises(ValueError):
        alns.iterate(One(), select, HillClimbing(), MaxIterations(1))


def test_zero_max_iterations():
    """
    Test that the algorithm return the initial solution when the
    stopping criterion is zero max iterations.
    """
    alns = get_alns_instance(
        [lambda state, rng: None], [lambda state, rng: None]
    )

    initial_solution = One()
    select = RouletteWheel([1, 1, 1, 1], 0.5, 1, 1)

    result = alns.iterate(
        initial_solution, select, HillClimbing(), MaxIterations(0)
    )

    assert_(result.best_state is initial_solution)


def test_zero_max_runtime():
    """
    Test that the algorithm return the initial solution when the
    stopping criterion is zero max runtime.
    """
    alns = get_alns_instance(
        [lambda state, rng: None], [lambda state, rng: None]
    )

    initial_solution = One()
    select = RouletteWheel([1, 1, 1, 1], 0.5, 1, 1)

    result = alns.iterate(
        initial_solution, select, HillClimbing(), MaxRuntime(0)
    )

    assert_(result.best_state is initial_solution)


def test_iterate_kwargs_are_correctly_passed_to_operators():
    def test_operator(state, rng, item):
        assert_(item is orig_item)
        return state

    alns = get_alns_instance([lambda state, rng, item: state], [test_operator])

    init_sol = One()
    select = RouletteWheel([1, 1, 1, 1], 0.5, 1, 1)
    orig_item = object()

    alns.iterate(
        init_sol, select, HillClimbing(), MaxIterations(10), item=orig_item
    )


def test_bugfix_pass_kwargs_to_on_best():
    """
    Exercises a bug where the on_best callback did not receive the kwargs
    passed to iterate().
    """

    def test_operator(state, rng, item):
        assert_(item is orig_item)
        return Zero()  # better, so on_best is triggered

    alns = get_alns_instance([lambda state, rng, item: state], [test_operator])
    alns.on_best(lambda state, rng, item: state)

    init_sol = One()
    select = RouletteWheel([1, 1, 1, 1], 0.5, 1, 1)
    orig_item = object()

    alns.iterate(
        init_sol, select, HillClimbing(), MaxIterations(10), item=orig_item
    )


# EXAMPLES --------------------------------------------------------------------


def test_trivial_example():
    """
    This tests the ALNS algorithm on a trivial example, where the initial
    solution is one, and any other operator returns zero.
    """
    alns = get_alns_instance(
        [lambda state, rng: Zero()], [lambda state, rng: Zero()]
    )

    select = RouletteWheel([1, 1, 1, 1], 0.5, 1, 1)
    result = alns.iterate(One(), select, HillClimbing(), MaxIterations(100))

    assert_equal(result.best_state.objective(), 0)


@mark.parametrize("seed,desired", [(0, 0.00995), (1, 0.02648), (2, 0.00981)])
def test_fixed_seed_outcomes(seed: int, desired: float):
    """
    Tests if fixing a seed results in deterministic outcomes even when using a
    'random' acceptance criterion (here SA).
    """
    alns = get_alns_instance(
        [lambda state, rng: VarObj(rng.random())],
        [lambda state, rng: None],
        seed,
    )

    select = RouletteWheel([1, 1, 1, 1], 0.5, 1, 1)
    sa = SimulatedAnnealing(1, 0.25, 1 / 100)

    result = alns.iterate(One(), select, sa, MaxIterations(100))
    assert_almost_equal(result.best_state.objective(), desired, decimal=5)


@mark.parametrize("max_iterations", [1, 10, 100])
def test_nonnegative_max_iterations(max_iterations):
    """
    Test that the result statistics have size equal to max iterations (+1).
    """
    alns = get_alns_instance(
        [lambda state, rng: Zero()], [lambda state, rng: Zero()]
    )

    initial_solution = One()
    select = RouletteWheel([1, 1, 1, 1], 0.5, 1, 1)

    result = alns.iterate(
        initial_solution,
        select,
        HillClimbing(),
        MaxIterations(max_iterations),
    )

    assert_equal(len(result.statistics.objectives), max_iterations + 1)
    assert_equal(len(result.statistics.runtimes), max_iterations)


@mark.parametrize("max_runtime", [0.01, 0.05, 0.1])
def test_nonnegative_max_runtime(max_runtime):
    """
    Test that the result runtime statistics match the stopping criterion.
    """
    alns = get_alns_instance(
        [lambda state, rng: Zero()], [lambda state, rng: Zero()]
    )

    initial_solution = One()
    select = RouletteWheel([1, 1, 1, 1], 0.5, 1, 1)

    result = alns.iterate(
        initial_solution, select, HillClimbing(), MaxRuntime(max_runtime)
    )

    assert_almost_equal(
        sum(result.statistics.runtimes), max_runtime, decimal=2
    )


# TODO test more complicated examples?
