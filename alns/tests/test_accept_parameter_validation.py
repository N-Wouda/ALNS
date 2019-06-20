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
        accept(Zero(), One(), -10, 0.5, state)


def test_raises_zero_temperature():
    """
    A zero temperature would result in a division by zero, which should not be
    allowed.
    """
    with assert_raises(ValueError):
        accept(Zero(), One(), 0, 0.5, state)


def test_raises_negative_decay_parameter():
    """
    A negative decay parameter would result in a negative temperature, which
    should not be allowed.
    """
    with assert_raises(ValueError):
        accept(Zero(), One(), 10, -0.5, state)


def test_raises_explosive_decay_parameter():
    """
    Temperatures would increase without bound for a decay parameter greater
    than one, so this should raise.
    """
    with assert_raises(ValueError):
        accept(Zero(), One(), 10, 2.5, state)


def test_raises_boundary_decay_parameters():
    """
    The boundary cases, zero and one, should both raise.
    """
    with assert_raises(ValueError):
        accept(Zero(), One(), 10, 0, state)

    with assert_raises(ValueError):
        accept(Zero(), One(), 10, 1, state)


def test_does_not_raise():
    """
    This set of parameters, on the other hand, should work correctly.
    """
    accept(Zero(), One(), 10, .5, state)
