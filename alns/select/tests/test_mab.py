from typing import List

import numpy as np
from mabwiser.mab import MAB, LearningPolicy
from numpy.testing import assert_equal, assert_raises
from pytest import mark

from alns.select import MABSelector
from alns.select.MABSelector import arm2ops, ops2arm


def make_dummy_mab(num_destroy, num_repair, op_coupling=None) -> MAB:
    arms = MABSelector.make_arms(num_destroy, num_repair, op_coupling)
    return MAB(arms=arms, learning_policy=LearningPolicy.EpsilonGreedy(0.15))


def make_bad_mab(num_destroy, num_repair, op_coupling=None) -> MAB:
    arms = ["foo", "bar", "quux"]
    return MAB(arms=arms, learning_policy=LearningPolicy.EpsilonGreedy(0.15))


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


@mark.parametrize(
    "num_destroy, num_repair, op_coupling",
    [
        (1, 1, None),
        (2, 2, None),
        (3, 1, None),
        (2, 2, np.array([[True, True], [True, True]])),
        (
            2,
            2,
            np.array(
                [
                    [
                        True,
                        False,
                    ],
                    [True, False],
                ]
            ),
        ),
    ],
)
def test_make_arms(num_destroy, num_repair, op_coupling):
    arms = MABSelector.make_arms(num_destroy, num_repair, op_coupling)
    op_coupling_sum = (
        np.invert(op_coupling).sum() if op_coupling is not None else 0
    )

    assert_equal(len(arms), num_destroy * num_repair - op_coupling_sum)

    for arm in arms:
        operators = arm2ops(arm)
        output = ops2arm(*operators)
        assert_equal(output, arm)


@mark.parametrize(
    "num_destroy, num_repair, op_coupling",
    [
        (0, 0, None),
        (2, 0, None),
        (0, 1, None),
        (1, 1, np.array([False])),
    ],
)
def test_make_arms_raises_value_error(num_destroy, num_repair, op_coupling):
    with assert_raises(ValueError):
        MABSelector.make_arms(num_destroy, num_repair, op_coupling)


def test_does_not_raise_valid_on_mab():
    mab = make_dummy_mab(1, 1)
    MABSelector([0, 0, 0, 0], mab, 1, 1)


@mark.parametrize(
    "scores, mab, num_destroy, num_repair",
    [
        ([5, 3, 2, -1], make_dummy_mab(1, 1), 1, 1),  # negative score
        ([5, 3, 2], make_dummy_mab(1, 1), 1, 1),  # len(score) < 4
        (
            [5, 3, 2, 1],
            make_bad_mab(1, 2),
            1,
            2,
        ),  # mab arms not made with _make_arms
    ],
)
def test_raises_invalid_arguments(
    scores: List[float],
    mab: MAB,
    num_destroy: int,
    num_repair: int,
):
    with assert_raises(ValueError):
        MABSelector(scores, mab, num_destroy, num_repair)


# TODO:
# tests that check the predictions of the mab
