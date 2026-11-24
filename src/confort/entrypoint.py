"""Self-registering, runnable configs."""

from __future__ import annotations

import dataclasses
from abc import ABCMeta
from typing import Any, Optional

from confort.config import InstantiateConfig
from confort.registry import registration_suppressed, registry


class EntrypointMeta(ABCMeta):
    """Registers every named :class:`Entrypoint` right after it is constructed.

    Registration happens here rather than in ``__post_init__`` so it still
    works when a subclass defines its own ``__post_init__``.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        obj = super().__call__(*args, **kwargs)
        name = getattr(obj, "_entrypoint", None)
        if name is not None and not registration_suppressed():
            registry.add(name, obj, help=obj._help)
        return obj


@dataclasses.dataclass
class Entrypoint(InstantiateConfig, metaclass=EntrypointMeta):
    """An :class:`InstantiateConfig` that can be launched from the CLI.

    Give an instance an ``_entrypoint`` name and it registers itself as a
    subcommand when created, i.e. when its module is imported. Its field values
    become the CLI defaults. Running it builds the target and calls its
    ``run()`` method::

        @dataclass
        class TrainerConfig(Entrypoint):
            lr: float = 1e-3

        @TrainerConfig.builds
        class Trainer:
            def __init__(self, config: TrainerConfig): ...
            def run(self): ...

        TrainerConfig(lr=1e-4, _entrypoint="train.fast", _help="Quick training")

    Instances without ``_entrypoint`` are ordinary configs and are not
    registered. Override :meth:`run` to change what "running" means.
    """

    _entrypoint: Optional[str] = dataclasses.field(default=None, kw_only=True, compare=False)
    _help: Optional[str] = dataclasses.field(default=None, kw_only=True, compare=False)

    def run(self, **kwargs: Any) -> Any:
        """Build the target with :meth:`setup` and call its ``run()`` method."""
        return self.setup(**kwargs).run()
