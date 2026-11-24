import sys

import pytest

import confort


@pytest.fixture(autouse=True)
def clean_registry():
    """Each test starts with an empty registry and un-imported fake project."""
    confort.registry.clear()
    for name in [m for m in sys.modules if m == "fakeproj" or m.startswith("fakeproj.")]:
        del sys.modules[name]
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    yield
    sys.path.pop(0)
    confort.registry.clear()


@pytest.fixture
def project():
    confort.discover("fakeproj", exclude=["fakeproj.broken"])
    return sys.modules["fakeproj.train"]
