from typing import List

from mabwiser.mab import LearningPolicy, NeighborhoodPolicy
from numpy.testing import assert_equal, assert_raises
from pytest import mark

from alns.select import MABSelector
from alns.select.MABSelector import arm2ops, ops2arm


@mark.parametrize(
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


def test_does_not_raise_on_valid_mab():
    MABSelector([0, 0, 0, 0], 2, 1, LearningPolicy.EpsilonGreedy(0.15))
    MABSelector(
        [0, 0, 0, 0],
        2,
        1,
        LearningPolicy.EpsilonGreedy(0.15),
        NeighborhoodPolicy.Radius(5),
    )
    MABSelector(
        [0, 0, 0, 0],
        2,
        1,
        LearningPolicy.EpsilonGreedy(0.15),
        NeighborhoodPolicy.Radius(5),
        1234567,
    )
    MABSelector(
        [0, 0, 0, 0], 2, 1, LearningPolicy.EpsilonGreedy(0.15), seed=1234567
    )


@mark.parametrize(
    "scores, learning_policy, num_destroy, num_repair",
    [
        (
            [5, 3, 2, -1],
            LearningPolicy.EpsilonGreedy(0.15),
            1,
            1,
        ),  # negative score
        (
            [5, 3, 2],
            LearningPolicy.EpsilonGreedy(0.15),
            1,
            1,
        ),  # len(score) < 4
    ],
)
def test_raises_invalid_arguments(
    scores: List[float],
    learning_policy: LearningPolicy,
    num_destroy: int,
    num_repair: int,
):
    with assert_raises(ValueError):
        MABSelector(scores, num_destroy, num_repair, learning_policy)


# TODO:
# tests that check the predictions of the mab
