from alns import ALNS
from numpy.testing import assert_equal


def test_add_destroy_operator():
    alns = ALNS()

    for count in [1, 2]:
        alns.add_destroy_operator(lambda item: None)
        assert_equal(len(alns.destroy_operators), count)


def test_add_repair_operator():
    alns = ALNS()

    for count in [1, 2]:
        alns.add_repair_operator(lambda item: None)
        assert_equal(len(alns.repair_operators), count)
