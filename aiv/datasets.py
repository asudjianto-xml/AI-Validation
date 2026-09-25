"""Access to data bundled with the package.

The credit-default example dataset ships inside the package so that examples run
without any external file. `default_data_path()` returns a real filesystem path
to it, resolving through importlib.resources so it works from an installed wheel,
an editable install, or a source checkout.
"""
from __future__ import annotations

from contextlib import ExitStack
from importlib.resources import as_file, files

_BUNDLED_CSV = "credit_default.csv"

# Keep extracted-file handles alive for the process lifetime. When the package is
# imported from a zip, as_file() materializes a temp copy that must not be removed
# while callers still hold the path.
_KEEPALIVE = ExitStack()


def default_data_path() -> str:
    """Filesystem path to the bundled credit-default CSV.

    Distributions that ship only the AML companion data omit this file; pass an
    explicit CSV path to the functions that accept one in that case.
    """
    resource = files("aiv.data").joinpath(_BUNDLED_CSV)
    if not resource.is_file():
        raise FileNotFoundError(
            f"{_BUNDLED_CSV} is not installed with this distribution of aiv; "
            "supply the path to a CSV file explicitly.")
    return str(_KEEPALIVE.enter_context(as_file(resource)))
