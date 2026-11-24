"""Build a tyro CLI from the registered entrypoints."""

from __future__ import annotations

import argparse
import contextvars
import dataclasses
import datetime
import pathlib
import sys
from typing import Annotated, Any, Iterable, Optional, Sequence, Union

import tyro
from tyro.constructors import ConstructorRegistry, StructConstructorSpec, StructTypeInfo

from confit.config import Config, load_file
from confit.discovery import discover as _discover
from confit.entrypoint import Entrypoint
from confit.registry import Registration, Registry, registry as _default_registry, suppress_registration

DEFAULT_MARKERS: tuple[Any, ...] = (tyro.conf.SuppressFixed, tyro.conf.FlagConversionOff)
"""tyro markers applied by default (the pgp setup): hide fixed fields, and
spell booleans as ``--flag True`` / ``--flag False``."""

# --- hide private fields from the CLI ------------------------------------------

_tyro_registry = ConstructorRegistry()
_inside_rule: contextvars.ContextVar[bool] = contextvars.ContextVar("confit_inside_rule", default=False)


@_tyro_registry.struct_rule
def _hide_private_fields(info: StructTypeInfo) -> Optional[StructConstructorSpec]:
    if _inside_rule.get() or not (isinstance(info.type, type) and issubclass(info.type, Config)):
        return None
    token = _inside_rule.set(True)
    try:  # Ask tyro's built-in dataclass rule for the spec, then patch it.
        spec = ConstructorRegistry.get_struct_spec(info)
    finally:
        _inside_rule.reset(token)
    if not isinstance(spec, StructConstructorSpec):
        return None
    fields = tuple(
        dataclasses.replace(f, type=Annotated[f.type, tyro.conf.Suppress]) if f.name.startswith("_") else f
        for f in spec.fields
    )
    return dataclasses.replace(spec, fields=fields)


# --- public API ----------------------------------------------------------------


def cli(
    discover: Union[str, Iterable[str], None] = None,
    *,
    exclude: Iterable[str] = (),
    args: Optional[Sequence[str]] = None,
    prog: Optional[str] = None,
    description: Optional[str] = None,
    run: bool = True,
    save_config: Union[str, pathlib.Path, None] = None,
    config_file_flag: Optional[str] = "--config-file",
    accent_color: Optional[str] = None,
    markers: Sequence[Any] = DEFAULT_MARKERS,
    registry: Optional[Registry] = None,
) -> Any:
    """Parse the command line and launch the chosen entrypoint.

    Args:
        discover: Package(s) to import recursively before building the CLI, so
            the entrypoints they define register themselves (see
            :func:`confit.discover`). ``None`` uses only what is already
            registered.
        exclude: Module names/patterns to skip during discovery.
        args: Arguments to parse; defaults to ``sys.argv[1:]``.
        prog: Program name shown in the help.
        description: Text shown at the top of the help.
        run: If true, call ``config.run()`` and return its result; if false,
            return the parsed :class:`~confit.Entrypoint` config instead.
            Function entrypoints are always called.
        save_config: If set, the parsed config is saved here before running.
            ``{name}`` and ``{time}`` are replaced by the entrypoint name and a
            ``YYYY-MM-DD_HH-MM-SS`` timestamp, e.g.
            ``"outputs/{name}/{time}/config.yaml"``.
        config_file_flag: Name of the flag that loads values from a YAML/JSON
            file before applying the other flags. ``None`` disables it.
        accent_color: tyro help accent color, e.g. ``"bright_yellow"``.
        markers: tyro markers applied to the whole CLI.
        registry: Registry to read entrypoints from (default: the global one).

    Returns:
        The return value of ``run()``, the parsed config (``run=False``), or
        the return value of a function entrypoint.
    """
    if discover is not None:
        _discover(discover, exclude)
    reg = registry if registry is not None else _default_registry
    argv = list(sys.argv[1:] if args is None else args)

    overrides: dict[str, Entrypoint] = {}
    if config_file_flag is not None:
        argv, path = _pop_flag(argv, config_file_flag)
        if path is not None:
            name, config = _load_config_file(reg, argv, path, config_file_flag)
            if name not in argv:
                argv.insert(0, name)
            overrides[name] = config
        if description is None or config_file_flag not in description:
            hint = f"Pass `{config_file_flag} PATH` to load values from a saved YAML/JSON config."
            description = f"{description}\n\n{hint}" if description else hint

    cli_type = _cli_type(reg, overrides)
    if len(reg) == 1 and argv and argv[0] == reg.names()[0]:
        argv = argv[1:]  # a single entrypoint has no subcommand; accept its name anyway

    if accent_color is not None:
        tyro.extras.set_accent_color(accent_color)
    with _tyro_registry, suppress_registration():
        result = tyro.cli(cli_type, args=argv, prog=prog, description=description, config=tuple(markers))

    if not isinstance(result, Entrypoint):
        return result  # a function entrypoint, already called by tyro
    if save_config is not None:
        path = str(save_config).format(
            name=result._entrypoint, time=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        result.save(path)
    return result.run() if run else result


def get_parser(
    discover: Union[str, Iterable[str], None] = None,
    *,
    exclude: Iterable[str] = (),
    prog: Optional[str] = None,
    description: Optional[str] = None,
    markers: Sequence[Any] = DEFAULT_MARKERS,
    registry: Optional[Registry] = None,
) -> argparse.ArgumentParser:
    """Return the ``argparse`` parser behind :func:`cli` (e.g. for sphinx-argparse docs)."""
    if discover is not None:
        _discover(discover, exclude)
    reg = registry if registry is not None else _default_registry
    with _tyro_registry:
        return tyro.extras.get_parser(
            _cli_type(reg, {}), prog=prog, description=description, config=tuple(markers)
        )


# --- internals -----------------------------------------------------------------


def _cli_type(reg: Registry, overrides: dict[str, Entrypoint]) -> Any:
    if len(reg) == 0:
        raise RuntimeError(
            "No entrypoints registered. Create an Entrypoint with `_entrypoint=...`, decorate a "
            "function with @confit.command, or pass `discover=` to import the modules defining them."
        )
    options = [_subcommand(r, overrides.get(r.name)) for r in reg]
    return options[0] if len(options) == 1 else Union[tuple(options)]  # type: ignore[valid-type]


def _subcommand(reg: Registration, override: Optional[Entrypoint]) -> Any:
    if reg.is_function:
        return Annotated[reg.target, tyro.conf.subcommand(name=reg.name, description=reg.help)]
    default = override if override is not None else reg.target
    return Annotated[
        type(default),
        tyro.conf.subcommand(name=reg.name, default=default, description=reg.help),
    ]


def _pop_flag(argv: list[str], flag: str) -> tuple[list[str], Optional[str]]:
    out, value, i = [], None, 0
    while i < len(argv):
        arg = argv[i]
        if arg == flag:
            if i + 1 >= len(argv):
                raise SystemExit(f"error: {flag} expects a path")
            value, i = argv[i + 1], i + 2
            continue
        if arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
        else:
            out.append(arg)
        i += 1
    return out, value


def _load_config_file(reg: Registry, argv: list[str], path: str, flag: str) -> tuple[str, Entrypoint]:
    data = load_file(path)
    name = next((a for a in argv if a in reg), None) or data.get("_entrypoint")
    if name is None and len(reg) == 1:
        name = reg.names()[0]
    if name is None:
        raise SystemExit(f"error: {path} does not name an entrypoint; pass the subcommand before {flag}")
    if name not in reg:
        raise SystemExit(f"error: entrypoint {name!r} from {path} is not registered")
    target = reg[name].target
    if not isinstance(target, Entrypoint):
        raise SystemExit(f"error: {flag} only works with config entrypoints, {name!r} is a function")
    return name, target.merge(data)
