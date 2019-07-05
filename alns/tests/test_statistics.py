from numpy.testing import assert_

from alns.Statistics import Statistics


def test_empty_new_statistics():
    """
    Tests if a new Statistics object starts empty.
    """
    statistics = Statistics()
    assert_(len(statistics.objectives) == 0)


def test_collect_objectives():
    """
    Tests if a Statistics object properly collects objective values.
    """
    statistics = Statistics()

    for objective in range(1, 100):
        statistics.collect_objective(objective)

        assert_(len(statistics.objectives) == objective)
        assert_(statistics.objectives[-1] == objective)
