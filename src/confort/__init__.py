"""confort: dataclass configs, self-registering entrypoints and a tyro CLI."""

from confort.cli import DEFAULT_MARKERS, cli, cli_type, tyro_registry
from confort.config import Config, FactoryConfig, InstantiateConfig, default
from confort.discovery import discover
from confort.entrypoint import Entrypoint
from confort.registry import Registration, Registry, command, registry, suppress_registration

__version__ = "0.1.0"

__all__ = [
    "Config",
    "InstantiateConfig",
    "FactoryConfig",
    "Entrypoint",
    "default",
    "command",
    "cli",
    "cli_type",
    "tyro_registry",
    "discover",
    "registry",
    "Registry",
    "Registration",
    "suppress_registration",
    "DEFAULT_MARKERS",
]
