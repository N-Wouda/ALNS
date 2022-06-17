import numpy.random as rnd
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from alns.accept import LateAcceptanceHillClimbing
from alns.tests.states import Zero, One, Two


@mark.parametrize("n_iterations", [-0.01, -10, 1.5])
def test_raises_invalid_n_iterations(n_iterations):
    with assert_raises(ValueError):
        LateAcceptanceHillClimbing(n_iterations=n_iterations)


@mark.parametrize(
    "n_iterations, on_improve, only_better",
    [
        (1, True, True),
        (10, False, True),
        (100, True, False),
        (1000, False, False),
    ],
)
def test_properties(n_iterations, on_improve, only_better):
    lahc = LateAcceptanceHillClimbing(n_iterations, on_improve, only_better)

    assert_equal(lahc.n_iterations, n_iterations)
    assert_equal(lahc.only_better, only_better)
    assert_equal(lahc.on_improve, on_improve)


@mark.parametrize("n_iterations", [3, 10, 50])
def test_lahc_accept(n_iterations):
    """
    Tests if LAHC accepts a solution that is better than the current solution
    n_iterations iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(n_iterations, False, False)

    for _ in range(n_iterations):
        assert_(lahc(rnd.RandomState(), Zero(), Two(), One()))

    # The previous current solution n_iterations ago has value 2, so the
    # candidate solution with value 1 should be accepted.
    assert_(lahc(rnd.RandomState(), Zero(), Zero(), One()))


@mark.parametrize("n_iterations", [3, 10, 50])
def test_lahc_reject(n_iterations):
    """
    Tests if LAHC rejects a solution that is worse than the current solution
    n_iterations iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(n_iterations, False, False)

    for _ in range(n_iterations):
        assert_(lahc(rnd.RandomState(), Zero(), One(), Zero()))

    # The previous current solution n_iterations ago has value 1, so the
    # candidate solution with value 1 should be rejected.
    assert_(not lahc(rnd.RandomState(), Zero(), Two(), One()))


@mark.parametrize("n_iterations", [3, 10, 50])
def test_lahc_on_improve_accept(n_iterations):
    """
    Tests if LAHC criterion with on_improve=True accepts a solution that
    is better than the current solution despite being worse than the
    previous current solution from n_iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(n_iterations, True, False)

    for _ in range(n_iterations):
        assert_(not lahc(rnd.RandomState(), Zero(), One(), Two()))

    # The candidate (1) is better than the current (2), hence it is accepted
    # despite being worse than the previous current (1).
    assert_(lahc(rnd.RandomState(), Zero(), Two(), One()))


@mark.parametrize("n_iterations", [3, 10, 50])
def test_lahc_only_better_reject(n_iterations):
    """
    Tests if LAHC criterion with only_better=True rejects a solution that
    is better than the previous current solution from n_iterations ago, because
    that previous current solution was not better than the current solution
    from 2 * n_iterations ago.
    """
    lahc = LateAcceptanceHillClimbing(n_iterations, False, True)

    for _ in range(n_iterations):
        assert_(not lahc(rnd.RandomState(), Zero(), One(), Two()))

    for _ in range(n_iterations):
        # The current solutions are not stored because they are worse
        # than the previous current solutions
        assert_(not lahc(rnd.RandomState(), Zero(), Two(), Two()))

    assert_(not lahc(rnd.RandomState(), Zero(), Two(), One()))


def test_update():
    """
    Test the _update method.
    """
    lahc = LateAcceptanceHillClimbing(10, False, False)

    assert_equal(lahc._update(rnd.RandomState(), Zero(), One(), Two()), 1)
    assert_equal(lahc._update(rnd.RandomState(), Zero(), Two(), One()), 2)


def test_update_only_better():
    """
    Test the _update method with only_better=True.
    """
    lahc = LateAcceptanceHillClimbing(1, False, True)

    # Ensure that the previous current is stored
    assert_(not lahc(rnd.RandomState(), Zero(), One(), Two()))

    # Previous current (1) is not updated because current is worse (2)
    assert_equal(lahc._update(rnd.RandomState(), Zero(), Two(), Two()), 1)
