"""The global registry of named entrypoints.

Entrypoints are added here automatically when an :class:`~confit.Entrypoint`
is created with an ``_entrypoint`` name, or when a function is decorated with
:func:`~confit.command`. The CLI reads this registry to build its subcommands.
"""

from __future__ import annotations

import contextlib
import contextvars
import sys
from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterator, Optional

_suppressed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "confit_registration_suppressed", default=False
)


@contextlib.contextmanager
def suppress_registration() -> Generator[None, None, None]:
    """Context manager: entrypoints created inside it are *not* registered.

    confit uses it internally whenever it makes copies of a registered config
    (CLI parsing, :meth:`Config.replace`, :meth:`Config.merge`), so copies never
    collide with the original. You rarely need it yourself.
    """
    token = _suppressed.set(True)
    try:
        yield
    finally:
        _suppressed.reset(token)


def registration_suppressed() -> bool:
    return _suppressed.get()


@dataclass(frozen=True)
class Registration:
    """One registered entrypoint.

    Attributes:
        name: The subcommand name, e.g. ``"train.closer"``.
        target: An :class:`~confit.Entrypoint` instance (its values become the
            CLI defaults) or a plain function (its signature becomes the CLI).
        help: Help text shown in the CLI. ``None`` falls back to the
            docstring of the config class or function.
        module: The module that registered it, used in error messages.
    """

    name: str
    target: Any
    help: Optional[str] = None
    module: Optional[str] = None

    @property
    def is_function(self) -> bool:
        from confit.entrypoint import Entrypoint

        return not isinstance(self.target, Entrypoint)


class Registry:
    """An ordered mapping from entrypoint names to :class:`Registration`."""

    def __init__(self) -> None:
        self._entries: dict[str, Registration] = {}

    def add(
        self,
        name: str,
        target: Any,
        help: Optional[str] = None,
        module: Optional[str] = None,
    ) -> Registration:
        """Register ``target`` under ``name``.

        Registering the same name twice is allowed only if the target is
        equivalent (this happens when a module is imported twice, e.g. once as
        ``__main__``); otherwise a :class:`ValueError` is raised.
        """
        if not isinstance(name, str) or not name or any(c.isspace() for c in name):
            raise ValueError(f"Invalid entrypoint name {name!r}: use a non-empty string without spaces.")
        module = module or _caller_module()
        new = Registration(name=name, target=target, help=help, module=module)
        old = self._entries.get(name)
        if old is not None:
            if _equivalent(old.target, target):
                return old
            raise ValueError(
                f"Entrypoint {name!r} is already registered (in module {old.module!r}); "
                f"cannot register a different one from module {module!r}. "
                "Pick another name, or use cfg.replace(...) to derive a variant."
            )
        self._entries[name] = new
        return new

    def remove(self, name: str) -> None:
        """Unregister ``name`` (raises :class:`KeyError` if missing)."""
        del self._entries[name]

    def clear(self) -> None:
        """Remove every registration."""
        self._entries.clear()

    def names(self) -> list[str]:
        return list(self._entries)

    def __getitem__(self, name: str) -> Registration:
        return self._entries[name]

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[Registration]:
        return iter(list(self._entries.values()))

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self.names()})"


registry = Registry()
"""The default, process-wide registry used by :func:`confit.cli`."""


def command(
    name: Optional[str] = None, *, help: Optional[str] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a plain function as a CLI subcommand.

    The function's parameters become CLI flags (via tyro); it is called with
    the parsed values and its return value is returned by :func:`confit.cli`.

    Args:
        name: Subcommand name. Defaults to the function name with ``_``
            replaced by ``-``.
        help: Help text. Defaults to the function docstring.

    Example::

        @confit.command("data.stats")
        def stats(root: str, verbose: bool = False) -> None:
            ...
    """
    if callable(name):  # used bare, as @command without parentheses
        return command()(name)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        registry.add(name or fn.__name__.replace("_", "-"), fn, help=help, module=fn.__module__)
        return fn

    return decorator


def _equivalent(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if callable(a) and callable(b) and not hasattr(a, "__dataclass_fields__"):
        return (getattr(a, "__module__", None), getattr(a, "__qualname__", None)) == (
            getattr(b, "__module__", None),
            getattr(b, "__qualname__", None),
        )
    try:
        return type(a).__qualname__ == type(b).__qualname__ and bool(a == b)
    except Exception:
        return False


def _caller_module() -> Optional[str]:
    """Name of the first module up the stack that is not confit or dataclasses."""
    frame = sys._getframe(1)
    while frame is not None:
        mod = frame.f_globals.get("__name__", "")
        if not (mod == "confit" or mod.startswith("confit.") or mod in ("dataclasses", "abc")):
            return mod
        frame = frame.f_back
    return None
