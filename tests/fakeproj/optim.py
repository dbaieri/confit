import collections
from dataclasses import dataclass
from types import ModuleType

from confit import FactoryConfig


@dataclass
class CounterConfig(FactoryConfig):
    """Builds a collections.Counter, standing in for e.g. torch.optim.Adam."""

    _module: ModuleType = collections
    _name: str = "Counter"
    a: int = 1
    b: int = 2
