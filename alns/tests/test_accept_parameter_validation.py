import numpy.random as rnd
from numpy.testing import assert_raises

from alns.accept import accept
from .states import Zero, One

state = rnd.RandomState(0)


def test_raises_negative_temperature():
    """
    A negative initial temperature should not be allowed.
    """
    with assert_raises(ValueError):
        accept(Zero(), One(), -10, state)


def test_raises_zero_temperature():
    """
    A zero temperature would result in a division by zero, which should not be
    allowed.
    """
    with assert_raises(ValueError):
        accept(Zero(), One(), 0, state)


def test_does_not_raise():
    """
    This set of parameters, on the other hand, should work correctly.
    """
    accept(Zero(), One(), 10, state)
