from typing import List

import numpy.random as rnd
import pytest
from numpy.testing import assert_, assert_equal, assert_raises

from alns.Outcome import Outcome
from alns.select import MABSelector
from alns.select.MABSelector import MABWISER_AVAILABLE, arm2ops, ops2arm
from alns.tests.states import Zero, ZeroWithOneContext, ZeroWithZeroContext

if MABWISER_AVAILABLE:
    from mabwiser.mab import LearningPolicy, NeighborhoodPolicy


@pytest.mark.skipif(not MABWISER_AVAILABLE, reason="MABWiser not available")
@pytest.mark.parametrize(
    "destroy_idx, repair_idx",
    [
        (0, 0),
        (0, 1),
        (3, 3),
        (12, 7),
        (0, 14),
    ],
)
def test_arm_conversion(destroy_idx, repair_idx):
    expected = (destroy_idx, repair_idx)
    actual = arm2ops(ops2arm(destroy_idx, repair_idx))

    assert_equal(actual, expected)


@pytest.mark.skipif(not MABWISER_AVAILABLE, reason="MABWiser not available")
def test_does_not_raise_on_valid_mab():
    policy = LearningPolicy.EpsilonGreedy(0.15)
    select = MABSelector([5, 0, 3, 0], 2, 1, policy)
    assert_equal(select.scores, [5, 0, 3, 0])
    assert_(len(select.mab.arms), 2)

    MABSelector([0, 0, 0, 0], 2, 1, policy, NeighborhoodPolicy.Radius(5))
    MABSelector(
        [1, 0, 0, 0], 2, 1, policy, NeighborhoodPolicy.Radius(5), 1234567
    )
    MABSelector([2, 1, 0, 0], 2, 1, policy, seed=1234567)


@pytest.mark.skipif(not MABWISER_AVAILABLE, reason="MABWiser not available")
@pytest.mark.parametrize(
    "scores, num_destroy, num_repair",
    [
        ([5, 3, 2, -1], 1, 1),  # negative score
        ([5, 3, 2], 1, 1),  # len(score) < 4
    ],
)
def test_raises_invalid_arguments(
    scores: List[float],
    num_destroy: int,
    num_repair: int,
):
    policy = LearningPolicy.EpsilonGreedy(0.15)
    with assert_raises(ValueError):
        MABSelector(scores, num_destroy, num_repair, policy)


@pytest.mark.skipif(not MABWISER_AVAILABLE, reason="MABWiser not available")
def test_call_with_only_one_operator_pair():
    # Only one operator pair, so the algorithm should select (0, 0).
    select = MABSelector(
        [2, 1, 1, 0], 1, 1, LearningPolicy.EpsilonGreedy(0.15)
    )
    rng = rnd.default_rng()

    for _ in range(10):
        selected = select(rng, Zero(), Zero())
        assert_equal(selected, (0, 0))


@pytest.mark.skipif(not MABWISER_AVAILABLE, reason="MABWiser not available")
def test_mab_epsilon_greedy():
    rng = rnd.default_rng()

    # epsilon=0 is equivalent to greedy selection
    select = MABSelector([2, 1, 1, 0], 2, 1, LearningPolicy.EpsilonGreedy(0.0))

    select.update(Zero(), 0, 0, outcome=Outcome.BETTER)
    selected = select(rng, Zero(), Zero())
    for _ in range(10):
        selected = select(rng, Zero(), Zero())
        assert_equal(selected, (0, 0))

    select.update(Zero(), 1, 0, outcome=Outcome.BEST)
    for _ in range(10):
        selected = select(rng, Zero(), Zero())
        assert_equal(selected, (1, 0))


@pytest.mark.skipif(not MABWISER_AVAILABLE, reason="MABWiser not available")
@pytest.mark.parametrize("alpha", [0.25, 0.5])
def test_mab_ucb1(alpha):
    rng = rnd.default_rng()
    select = MABSelector([2, 1, 1, 0], 2, 1, LearningPolicy.UCB1(alpha))

    select.update(Zero(), 0, 0, outcome=Outcome.BEST)
    mab_select = select(rng, Zero(), Zero())
    assert_equal(mab_select, (0, 0))

    select.update(Zero(), 0, 0, outcome=Outcome.REJECT)
    mab_select = select(rng, Zero(), Zero())
    assert_equal(mab_select, (0, 0))


@pytest.mark.skipif(not MABWISER_AVAILABLE, reason="MABWiser not available")
def test_contextual_mab_requires_context():
    select = MABSelector(
        [2, 1, 1, 0],
        2,
        1,
        LearningPolicy.LinGreedy(0),
    )
    # error: "Zero" state has no get_context method
    with assert_raises(AttributeError):
        select.update(Zero(), 0, 0, outcome=Outcome.BEST)


@pytest.mark.skipif(not MABWISER_AVAILABLE, reason="MABWiser not available")
def text_contextual_mab_uses_context():
    rng = rnd.default_rng()
    select = MABSelector(
        [2, 1, 1, 0],
        2,
        1,
        # epsilon=0 is equivalent to greedy
        LearningPolicy.LinGreedy(0),
    )

    select.update(ZeroWithZeroContext(), 0, 0, outcome=Outcome.REJECT)
    select.update(ZeroWithZeroContext(), 0, 0, outcome=Outcome.REJECT)
    select.update(ZeroWithZeroContext(), 1, 0, outcome=Outcome.BEST)

    select.update(ZeroWithOneContext(), 1, 0, outcome=Outcome.REJECT)
    select.update(ZeroWithOneContext(), 1, 0, outcome=Outcome.REJECT)
    select.update(ZeroWithOneContext(), 0, 0, outcome=Outcome.BEST)

    mab_select = select(rng, ZeroWithZeroContext(), ZeroWithZeroContext())
    assert_equal(mab_select, (1, 0))

    mab_select = select(rng, ZeroWithZeroContext(), ZeroWithZeroContext())
    assert_equal(mab_select, (0, 0))
