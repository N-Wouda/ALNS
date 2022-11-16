import sys

# TODO switch over to importlib.metadata once we drop support for Python 3.7
import pkg_resources  # type: ignore


def show_versions():
    """
    This function prints helpful debugging information that's useful when
    filing bug reports.
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
