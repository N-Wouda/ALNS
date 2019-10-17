from numpy.testing import assert_equal, assert_almost_equal

from alns.Statistics import Statistics


def test_empty_new_statistics():
    """
    Tests if a new Statistics object starts empty.
    """
    statistics = Statistics()

    assert_equal(len(statistics.objectives), 0)
    assert_equal(len(statistics.destroy_operator_counts), 0)
    assert_equal(len(statistics.repair_operator_counts), 0)


def test_collect_objectives():
    """
    Tests if a Statistics object properly collects objective values.
    """
    statistics = Statistics()

    for objective in range(1, 100):
        statistics.collect_objective(objective)

        assert_equal(len(statistics.objectives), objective)
        assert_almost_equal(statistics.objectives[-1], objective)


def test_collect_destroy_counts_example():
    """
    Tests if collecting for a destroy operator works as expected in a simple
    example.
    """
    statistics = Statistics()

    # This should twice collect for a method "destroy_test", at index 1, such
    # that the total count is [0, 2, 0, 0] for this operator.
    statistics.collect_destroy_operator("destroy_test", 1)
    statistics.collect_destroy_operator("destroy_test", 1)

    for idx, count in enumerate([0, 2, 0, 0]):
        assert_equal(statistics.destroy_operator_counts["destroy_test"][idx],
                     count)


def test_collect_repair_counts_example():
    """
    Tests if collecting for a repair operator works as expected in a simple
    example.
    """
    statistics = Statistics()

    # This should once collect for a method "repair_test", at index 2, such
    # that the total count is [0, 0, 1, 0] for this operator.
    statistics.collect_repair_operator("repair_test", 2)

    for idx, count in enumerate([0, 0, 1, 0]):
        assert_equal(statistics.repair_operator_counts["repair_test"][idx],
                     count)
