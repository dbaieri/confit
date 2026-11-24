"""Base config classes.

All configs are plain :mod:`dataclasses`. Subclass one of the bases below and
decorate the subclass with ``@dataclass``:

* :class:`Config` - printable, serialisable, with nested overrides.
* :class:`InstantiateConfig` - a config that builds an object: ``cfg.setup()``
  calls ``target(cfg)``.
* :class:`FactoryConfig` - a config that calls ``module.<name>(**fields)``,
  e.g. a ``torch.optim`` optimizer, without writing a wrapper class.

Fields whose name starts with ``_`` are *private*: they are hidden from the
CLI, from :meth:`Config.to_dict` and from :meth:`Config.__str__`.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import functools
import importlib
import json
import pathlib
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Type, TypeVar, Union

from confit.registry import suppress_registration

C = TypeVar("C", bound="Config")
T = TypeVar("T")


def default(value: T) -> T:
    """Use a mutable value (list, dict, nested config...) as a dataclass default.

    Shorthand for ``field(default_factory=lambda: copy.deepcopy(value))``, so
    every instance gets its own copy.

    Example::

        @dataclass
        class TrainerConfig(InstantiateConfig):
            milestones: list[int] = default([100, 200])
            optimizer: FactoryConfig = default(AdamConfig(lr=1e-4))
    """
    return dataclasses.field(default_factory=lambda: copy.deepcopy(value))


def _is_private(name: str) -> bool:
    return name.startswith("_")


@dataclasses.dataclass
class Config:
    """Base class for every confit config.

    Provides pretty printing (``str(cfg)``), conversion to plain dicts,
    saving/loading to YAML or JSON, and dotted-path overrides.
    """

    # --- inspection -----------------------------------------------------

    def __str__(self) -> str:
        lines = [type(self).__name__ + ":"]
        for f in dataclasses.fields(self):
            if _is_private(f.name):
                continue
            text = _format_value(getattr(self, f.name))
            first, *rest = text.split("\n")
            lines.append(f"{f.name}: {first}")
            lines.extend(rest)
        return "\n    ".join(lines)

    def to_dict(self, include_private: bool = False) -> dict[str, Any]:
        """Return the config as nested plain Python data (dicts, lists, scalars).

        Nested configs become dicts, tuples become lists, paths become strings
        and enums become their value. Private fields are skipped unless
        ``include_private`` is true. Values of other types are converted with
        ``str()``.
        """
        return {
            f.name: _to_plain(getattr(self, f.name), include_private)
            for f in dataclasses.fields(self)
            if include_private or not _is_private(f.name)
        }

    # --- persistence ----------------------------------------------------

    def save(self, path: Union[str, pathlib.Path]) -> pathlib.Path:
        """Write the config to ``path`` as YAML (``.yaml``/``.yml``) or JSON (``.json``).

        Parent directories are created. The file also records the config class
        (``_class``) and, for entrypoints, the entrypoint name
        (``_entrypoint``), so it can be passed back to the CLI with
        ``--config-file``. Returns the path written.
        """
        path = pathlib.Path(path)
        data: dict[str, Any] = {"_class": f"{type(self).__module__}.{type(self).__qualname__}"}
        name = getattr(self, "_entrypoint", None)
        if name is not None:
            data["_entrypoint"] = name
        data.update(self.to_dict())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump_text(data, path))
        return path

    def merge(self: C, source: Union[str, pathlib.Path, Mapping[str, Any]]) -> C:
        """Return a copy with values taken from a dict or a YAML/JSON file.

        Keys are matched against fields recursively; nested configs keep their
        class (so a file saved from a ``PatchDatasetConfig`` field restores into
        it). Keys starting with ``_`` are ignored, unknown keys raise
        :class:`KeyError`. Lists are converted back to tuples where the current
        value is a tuple.
        """
        data = load_file(source) if isinstance(source, (str, pathlib.Path)) else source
        with suppress_registration():
            return _merge(self, data, path=type(self).__name__)

    # --- overrides ------------------------------------------------------

    def replace(self: C, **overrides: Any) -> C:
        """Return a copy with some values changed; dotted keys reach nested configs.

        Example::

            small = train_closer.replace(**{"optimizer.lr": 1e-5, "batch_size": 8})

        For an :class:`~confit.Entrypoint`, the copy is registered as a new
        CLI command only if you pass a new ``_entrypoint`` name.
        """
        if "_entrypoint" in overrides:
            return _replace(self, overrides)
        with suppress_registration():
            return _replace(self, overrides)

    def __deepcopy__(self: C, memo: dict[int, Any]) -> C:
        # Modules (e.g. FactoryConfig._module) cannot be deep-copied; share them.
        new = copy.copy(self)
        memo[id(self)] = new
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, ModuleType):
                object.__setattr__(new, f.name, copy.deepcopy(value, memo))
        return new


@dataclasses.dataclass
class InstantiateConfig(Config):
    """A config that knows which class (or function) it builds.

    ``cfg.setup(**kwargs)`` returns ``target(cfg, **kwargs)``. The target can be
    given in two ways:

    * the pgp style, as a field default::

          @dataclass
          class TrainerConfig(InstantiateConfig):
              _target: Type = field(default_factory=lambda: Trainer)

    * with the :meth:`builds` decorator on the target class, which avoids the
      forward-reference lambda::

          @TrainerConfig.builds
          class Trainer:
              def __init__(self, config: TrainerConfig): ...

    The ``_target`` field wins over :meth:`builds` when both are set.
    """

    _target: Optional[Type] = dataclasses.field(default=None, kw_only=True)

    @classmethod
    def builds(cls, target: T) -> T:
        """Decorator: declare that this config class builds ``target``.

        Subclasses inherit the target unless they declare their own.
        Returns ``target`` unchanged.
        """
        if not callable(target):
            raise TypeError(f"{cls.__name__}.builds expects a class or function, got {target!r}")
        cls._confit_target_ = target  # type: ignore[attr-defined]
        return target

    @property
    def target(self) -> Callable[..., Any]:
        """The class or function :meth:`setup` will call."""
        target = self._target or getattr(type(self), "_confit_target_", None)
        if target is None:
            name = type(self).__name__
            raise TypeError(
                f"{name} has no target: set a `_target` default, or decorate the built "
                f"class with @{name}.builds"
            )
        return target

    def setup(self, **kwargs: Any) -> Any:
        """Build the target object: ``target(self, **kwargs)``."""
        return self.target(self, **kwargs)


@dataclasses.dataclass
class FactoryConfig(Config):
    """A config that calls an attribute of a module with the config's fields.

    ``cfg.setup(**kwargs)`` returns ``getattr(_module, _name)(**public_fields,
    **kwargs)``. ``_module`` may be a module object or an import path string,
    and ``_name`` may be dotted (``"lr_scheduler.StepLR"``).

    Example::

        @dataclass
        class AdamConfig(FactoryConfig):
            _module: ModuleType = torch.optim
            _name: str = "Adam"
            lr: float = 1e-3

        optimizer = AdamConfig(lr=1e-4).setup(params=model.parameters())
    """

    _name: Optional[str] = dataclasses.field(default=None, kw_only=True)
    _module: Union[ModuleType, str, None] = dataclasses.field(default=None, kw_only=True)

    @property
    def target(self) -> Callable[..., Any]:
        """The callable :meth:`setup` will call."""
        if self._name is None or self._module is None:
            raise TypeError(f"{type(self).__name__} needs both `_module` and `_name` to be set")
        module = importlib.import_module(self._module) if isinstance(self._module, str) else self._module
        return functools.reduce(getattr, self._name.split("."), module)

    def setup(self, **kwargs: Any) -> Any:
        """Call the target with every public field plus ``kwargs`` (which win)."""
        params = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if not _is_private(f.name)}
        params.update(kwargs)
        return self.target(**params)


# --- helpers ----------------------------------------------------------------


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)) and any(isinstance(v, Config) for v in value):
        return "[" + "\n".join(str(v) for v in value) + "]"
    return str(value)


def _to_plain(value: Any, include_private: bool) -> Any:
    if isinstance(value, Config):
        return value.to_dict(include_private)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _to_plain(getattr(value, f.name), include_private)
            for f in dataclasses.fields(value)
            if include_private or not _is_private(f.name)
        }
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v, include_private) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_plain(v, include_private) for v in value]
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, pathlib.PurePath):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, ModuleType):
        return value.__name__
    return str(value)


def _coerce(current: Any, new: Any) -> Any:
    """Convert a value loaded from a file to the type of the value it replaces."""
    if isinstance(current, tuple) and isinstance(new, list):
        return tuple(new)
    if isinstance(current, pathlib.PurePath) and isinstance(new, str):
        return type(current)(new)
    if isinstance(current, enum.Enum) and not isinstance(new, enum.Enum):
        return type(current)(new)
    if isinstance(current, float) and isinstance(new, int) and not isinstance(new, bool):
        return float(new)
    return new


def _merge(obj: Any, data: Mapping[str, Any], path: str) -> Any:
    names = {f.name for f in dataclasses.fields(obj) if f.init}
    changes: dict[str, Any] = {}
    for key, value in data.items():
        if _is_private(key):
            continue
        if key not in names:
            raise KeyError(f"Unknown config key {path}.{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and not isinstance(current, type) and isinstance(value, Mapping):
            changes[key] = _merge(current, value, f"{path}.{key}")
        else:
            changes[key] = _coerce(current, value)
    return dataclasses.replace(obj, **changes)


def _replace(obj: Any, overrides: Mapping[str, Any]) -> Any:
    direct: dict[str, Any] = {}
    nested: dict[str, dict[str, Any]] = {}
    for key, value in overrides.items():
        head, _, rest = key.partition(".")
        if rest:
            nested.setdefault(head, {})[rest] = value
        else:
            direct[head] = value
    names = {f.name for f in dataclasses.fields(obj) if f.init}
    for key in [*direct, *nested]:
        if key not in names:
            raise KeyError(f"{type(obj).__name__} has no field {key!r}")
    for head, sub in nested.items():
        if head in direct:
            raise ValueError(f"Cannot override both {head!r} and its nested fields in one call")
        child = getattr(obj, head)
        if not (dataclasses.is_dataclass(child) and not isinstance(child, type)):
            raise TypeError(f"{type(obj).__name__}.{head} is not a config, cannot set nested {list(sub)}")
        direct[head] = _replace(child, sub)
    return dataclasses.replace(obj, **direct)


def load_file(path: Union[str, pathlib.Path]) -> dict[str, Any]:
    """Read a YAML or JSON file (chosen by extension) into a dict."""
    path = pathlib.Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    elif path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config file extension {path.suffix!r}; use .yaml, .yml or .json")
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return data


def dump_text(data: Mapping[str, Any], path: pathlib.Path) -> str:
    if path.suffix.lower() == ".json":
        return json.dumps(data, indent=2) + "\n"
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        return yaml.safe_dump(dict(data), sort_keys=False)
    raise ValueError(f"Unsupported config file extension {path.suffix!r}; use .yaml, .yml or .json")
