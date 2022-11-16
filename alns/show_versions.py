import sys

# TODO switch over to importlib.metadata once we drop support for Python 3.7
import pkg_resources  # type: ignore


def show_versions():
    """
    This function prints version information that is useful when filing bug
    reports.

    Examples
    --------
    Calling this function should print information like the following
    (dependency versions in your local installation will likely differ):

    >>> import alns
    >>> alns.show_versions()
    INSTALLED VERSIONS
    ------------------
          alns: 5.0.1
         numpy: 1.23.4
    matplotlib: 3.5.1
        Python: 3.9.9
    """
    alns = pkg_resources.get_distribution("alns")
    numpy = pkg_resources.get_distribution("numpy")
    matplotlib = pkg_resources.get_distribution("matplotlib")
    python_version = ".".join(map(str, sys.version_info[:3]))

    print("INSTALLED VERSIONS")
    print("------------------")
    print(f"      alns: {alns.version}")
    print(f"     numpy: {numpy.version}")
    print(f"matplotlib: {matplotlib.version}")
    print(f"    Python: {python_version}")
