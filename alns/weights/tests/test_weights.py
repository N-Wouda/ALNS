from numpy.testing import assert_warns


def test_alns_weights_warns_deprecation():
    with assert_warns(DeprecationWarning):
        import alns.weights  # noqa
