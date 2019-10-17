import pytest
from matplotlib import pyplot as plt


@pytest.fixture(autouse=True)
def auto_close_all_figures(request):
    """
    Helper method that closes active testing figures in-between tests. See
    e.g. https://github.com/matplotlib/matplotlib/issues/15079 for details.
    """
    if "matplotlib" in request.keywords:
        plt.close("test")
        plt.close("reference")
