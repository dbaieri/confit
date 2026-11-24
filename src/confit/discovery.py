"""Import every module of a package so its entrypoints register themselves."""

from __future__ import annotations

import fnmatch
import importlib
import pkgutil
from typing import Iterable, Union


def discover(
    packages: Union[str, Iterable[str]],
    exclude: Iterable[str] = (),
) -> list[str]:
    """Recursively import ``packages`` and all their submodules.

    Args:
        packages: A package/module name or a list of them (e.g. ``"pgp"``).
        exclude: Module names or glob patterns to skip, matched against the
            full dotted name. Excluding a package also skips everything inside
            it (``"pgp.utils"`` skips ``pgp.utils.geometry``). Modules named
            ``__main__`` are always skipped.

    Returns:
        The names of the modules that were imported, in import order.

    Import errors are not swallowed: a broken module fails loudly.
    """
    if isinstance(packages, str):
        packages = [packages]
    patterns = list(exclude)
    imported: list[str] = []

    def excluded(name: str) -> bool:
        return name.rsplit(".", 1)[-1] == "__main__" or any(
            fnmatch.fnmatchcase(name, p) or name.startswith(p + ".") for p in patterns
        )

    def visit(name: str) -> None:
        if excluded(name):
            return
        module = importlib.import_module(name)
        imported.append(name)
        for info in pkgutil.iter_modules(getattr(module, "__path__", None) or []):
            visit(f"{name}.{info.name}")

    for package in packages:
        visit(package)
    return imported
