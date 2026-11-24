"""confit: dataclass configs, self-registering entrypoints and a tyro CLI."""

from confit.cli import DEFAULT_MARKERS, cli, get_parser
from confit.config import Config, FactoryConfig, InstantiateConfig, default
from confit.discovery import discover
from confit.entrypoint import Entrypoint
from confit.registry import Registration, Registry, command, registry, suppress_registration

__version__ = "0.1.0"

__all__ = [
    "Config",
    "InstantiateConfig",
    "FactoryConfig",
    "Entrypoint",
    "default",
    "command",
    "cli",
    "get_parser",
    "discover",
    "registry",
    "Registry",
    "Registration",
    "suppress_registration",
    "DEFAULT_MARKERS",
]
