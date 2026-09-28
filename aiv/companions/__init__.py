"""Chapter companion notebooks and the synthetic AML evidence they read.

The evidence records name their files by paths relative to one data root, for
example ``book/evidence/aml_medoid_loop_leaf_output_20260921/results.json``, and
they carry the SHA-256 digest of each file. The data root therefore keeps that
relative layout wherever it lives, so every recorded path and digest resolves
unchanged. It holds evidence, helper modules and notebooks only.

The data root is found in this order:

1. the directory named by the ``MODEL_VALIDATION_ROOT`` environment variable;
2. the repository checkout that contains this package, for an editable install;
3. the copy installed with this package (``aiv/companions/_data``).

``aiv-companions copy DEST`` copies the notebooks to a writable directory. They
locate the data root themselves when they run, so they may be opened anywhere.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ENVIRONMENT_VARIABLE = "MODEL_VALIDATION_ROOT"
MARKER = "book/notebooks/regional_case/companion.py"
NOTEBOOK_SETS = {
    "regional_case": "book/notebooks/regional_case",
    "regional_live": "book/notebooks/regional_live",
}


def is_data_root(path) -> bool:
    return (Path(path) / MARKER).is_file()


def locate() -> tuple[Path, str]:
    """Return the data root and a short description of where it was found."""
    configured = os.environ.get(ENVIRONMENT_VARIABLE)
    if configured:
        root = Path(configured).expanduser().resolve()
        if not is_data_root(root):
            raise FileNotFoundError(
                f"{ENVIRONMENT_VARIABLE}={configured} does not contain {MARKER}")
        return root, ENVIRONMENT_VARIABLE
    # A checkout comes first so that a staged copy never shadows the files it was built from.
    checkout = Path(__file__).resolve().parents[2]
    if is_data_root(checkout):
        return checkout, "repository checkout"
    installed = Path(__file__).resolve().parent / "_data"
    if is_data_root(installed):
        return installed, "installed aiv package"
    raise FileNotFoundError(
        "Companion data not found. Install the aiv package from a built wheel, "
        f"or set {ENVIRONMENT_VARIABLE} to a directory containing {MARKER}.")


def data_root() -> Path:
    return locate()[0]


def copy_notebooks(destination, overwrite: bool = False) -> list[Path]:
    """Copy both notebook sets into ``destination``; return the copied files."""
    root, destination = data_root(), Path(destination).expanduser().resolve()
    copied = []
    for name, relative in NOTEBOOK_SETS.items():
        target = destination / name
        target.mkdir(parents=True, exist_ok=True)
        for source in sorted((root / relative).glob("ch[0-9][0-9]_*.ipynb")):
            path = target / source.name
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
            shutil.copyfile(source, path)
            copied.append(path)
    return copied


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="aiv-companions", description="Locate or copy the chapter companion notebooks.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("where", help="print the data root and how it was found")
    copy = commands.add_parser("copy", help="copy the notebooks to a writable directory")
    copy.add_argument("destination")
    copy.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.command == "where":
        root, source = locate()
        print(f"{root}  ({source})")
    else:
        for path in copy_notebooks(arguments.destination, arguments.overwrite):
            print(path)


if __name__ == "__main__":
    main()
