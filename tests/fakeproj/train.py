from dataclasses import dataclass, field
from typing import Tuple, Type

from confit import Entrypoint, FactoryConfig, default

from .optim import CounterConfig


@dataclass
class TrainerConfig(Entrypoint):
    """Train a fake model."""

    lr: float = 1e-3
    steps: int = 10
    amp: bool = False
    milestones: Tuple[int, ...] = (5, 8)
    tags: list = default(["a"])
    optimizer: FactoryConfig = default(CounterConfig())


@TrainerConfig.builds
class Trainer:
    def __init__(self, config: TrainerConfig, extra=None):
        self.config = config
        self.extra = extra

    def run(self):
        return ("trained", self.config.lr, self.config.steps, self.config.amp)


@dataclass
class LegacyConfig(Entrypoint):
    """Uses the pgp-style _target field."""

    _target: Type = field(default_factory=lambda: Legacy)
    x: int = 0


class Legacy:
    def __init__(self, config):
        self.config = config

    def run(self):
        return ("legacy", self.config.x)


train_fast = TrainerConfig(lr=1e-2, steps=3, _entrypoint="train.fast", _help="Quick run")
train_long = TrainerConfig(steps=1000, _entrypoint="train.long")
legacy = LegacyConfig(x=7, _entrypoint="legacy")
unnamed = TrainerConfig(steps=1)  # not registered
