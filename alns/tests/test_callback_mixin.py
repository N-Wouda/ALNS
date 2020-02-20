from numpy.testing import assert_, assert_no_warnings, assert_warns

from alns import CallbackFlag
from alns.CallbackMixin import CallbackMixin
from alns.exceptions_warnings import OverwriteWarning


def dummy_callback():
    return None


def test_insert_extraction_on_best():
    """
    Tests if regular add/return callback works for ON_BEST.
    """
    mixin = CallbackMixin()
    mixin.on_best(dummy_callback)

    assert_(mixin.has_callback(CallbackFlag.ON_BEST))
    assert_(mixin.callback(CallbackFlag.ON_BEST) is dummy_callback)


def test_overwrite_warns_on_best():
    """
    There can only be a single callback for each event point, so inserting two
    (or more) should warn the previous callback for ON_BEST is overwritten.
    """
    mixin = CallbackMixin()

    with assert_no_warnings():  # first insert is fine..
        mixin.on_best(dummy_callback)

    with assert_warns(OverwriteWarning):  # .. but second insert should warn
        mixin.on_best(dummy_callback)
