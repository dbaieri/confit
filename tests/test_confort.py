import collections
import dataclasses
import json
import fractions
from dataclasses import dataclass
from types import ModuleType

import pytest

import confort
from confort import Config, Entrypoint, FactoryConfig, InstantiateConfig, default


# --- configs -------------------------------------------------------------------


@dataclass
class Inner(Config):
    k: int = 3
    _secret: str = "hidden"


@dataclass
class Outer(Config):
    name: str = "x"
    inner: Inner = default(Inner())
    sizes: tuple = (1, 2)


def test_str_and_to_dict_hide_private_fields():
    cfg = Outer()
    assert cfg.to_dict() == {"name": "x", "inner": {"k": 3}, "sizes": [1, 2]}
    assert cfg.to_dict(include_private=True)["inner"]["_secret"] == "hidden"
    text = str(cfg)
    assert "Outer:" in text and "Inner:" in text and "_secret" not in text


def test_default_gives_independent_copies():
    a, b = Outer(), Outer()
    assert a.inner is not b.inner


def test_replace_dotted_and_validation():
    cfg = Outer().replace(**{"inner.k": 9, "name": "y"})
    assert (cfg.name, cfg.inner.k) == ("y", 9)
    with pytest.raises(KeyError):
        Outer().replace(nope=1)
    with pytest.raises(ValueError):
        Outer().replace(**{"inner": Inner(), "inner.k": 1})
    with pytest.raises(TypeError):
        Outer().replace(**{"name.k": 1})


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_save_and_merge_roundtrip(tmp_path, suffix):
    path = Outer(name="saved", inner=Inner(k=5), sizes=(4, 5)).save(tmp_path / f"cfg{suffix}")
    restored = Outer().merge(path)
    assert restored == Outer(name="saved", inner=Inner(k=5), sizes=(4, 5))
    assert isinstance(restored.sizes, tuple)
    if suffix == ".json":
        assert json.loads(path.read_text())["_class"].endswith("Outer")
    with pytest.raises(KeyError):
        Outer().merge({"inner": {"bad": 1}})


def test_instantiate_with_builds_and_target_field():
    @dataclass
    class ThingConfig(InstantiateConfig):
        v: int = 1

    @ThingConfig.builds
    class Thing:
        def __init__(self, config, extra=0):
            self.total = config.v + extra

    assert ThingConfig(v=2).setup(extra=3).total == 5

    @dataclass
    class OtherConfig(InstantiateConfig):
        _target: type = dataclasses.field(default_factory=lambda: Box)

    class Box:
        def __init__(self, config):
            self.config = config

    assert isinstance(OtherConfig().setup(), Box)

    with pytest.raises(TypeError, match="no target"):
        InstantiateConfig().setup()


def test_factory_config():
    @dataclass
    class FractionConfig(FactoryConfig):
        _module: ModuleType = fractions
        _name: str = "Fraction"
        numerator: int = 1
        denominator: int = 2

    assert FractionConfig().setup() == fractions.Fraction(1, 2)
    assert FractionConfig().setup(denominator=4) == fractions.Fraction(1, 4)

    @dataclass
    class DottedConfig(FactoryConfig):
        _module: str = "os"
        _name: str = "path.basename"
        p: str = "/a/b.txt"

    assert DottedConfig().setup() == "b.txt"


def test_deepcopy_with_module_field():
    import copy

    @dataclass
    class C(FactoryConfig):
        _module: ModuleType = collections
        _name: str = "Counter"
        items: list = default([1])

    c = C()
    d = copy.deepcopy(c)
    assert d._module is collections and d.items is not c.items


# --- registration ---------------------------------------------------------------


def test_discovery_registers_entrypoints(project):
    assert sorted(confort.registry.names()) == ["add-numbers", "legacy", "train.fast", "train.long"]
    assert confort.registry["train.fast"].help == "Quick run"
    assert confort.registry["train.fast"].module == "fakeproj.train"


def test_discovery_exclude_and_main():
    with pytest.raises(ImportError, match="excluded package"):
        confort.discover("fakeproj")
    imported = confort.discover("fakeproj", exclude=["fakeproj.broken"])
    assert "fakeproj.__main__" not in imported and "fakeproj.sub.tools" in imported


def test_copies_do_not_reregister(project):
    variant = project.train_fast.replace(lr=0.5)
    assert variant.lr == 0.5 and confort.registry["train.fast"].target.lr == 1e-2
    dataclasses.replace(project.train_fast)  # identical copy: silently ignored


def test_replace_with_new_name_registers(project):
    project.train_fast.replace(steps=99, _entrypoint="train.fast99")
    assert confort.registry["train.fast99"].target.steps == 99


def test_conflicting_names_raise(project):
    with pytest.raises(ValueError, match="already registered"):
        project.TrainerConfig(steps=1, _entrypoint="train.fast")


def test_registration_survives_custom_post_init():
    @dataclass
    class P(Entrypoint):
        x: int = 0

        def __post_init__(self):
            self.x += 1

    P(_entrypoint="p")
    assert confort.registry["p"].target.x == 1


def test_entrypoint_run(project):
    assert project.train_fast.run() == ("trained", 1e-2, 3, False)
    assert project.legacy.run() == ("legacy", 7)


# --- CLI -----------------------------------------------------------------------


def test_cli_runs_with_overrides(project):
    out = confort.cli(args=["train.fast", "--steps", "5", "--amp", "True", "--optimizer.a", "4"])
    assert out == ("trained", 1e-2, 5, True)
    cfg = confort.cli(args=["train.fast", "--optimizer.a", "4"], run=False)
    assert cfg.optimizer.a == 4 and isinstance(cfg.optimizer, project.CounterConfig)
    assert cfg.optimizer.setup() == collections.Counter(a=4, b=2)
    assert len(confort.registry) == 4  # parsing did not register copies


def test_cli_discover_argument():
    out = confort.cli("fakeproj", exclude=["fakeproj.broken"], args=["legacy", "--x", "1"])
    assert out == ("legacy", 1)


def test_cli_function_entrypoint(project):
    assert confort.cli(args=["add-numbers", "--a", "40"]) == 42


def test_cli_help_hides_private_fields(project, capsys):
    with pytest.raises(SystemExit):
        confort.cli(args=["train.fast", "--help"])
    out = capsys.readouterr().out
    assert "--steps" in out and "--optimizer.a" in out
    assert "_entrypoint" not in out and "_name" not in out and "_target" not in out


def test_cli_config_file_and_save(project, tmp_path):
    template = str(tmp_path / "{name}" / "config.yaml")
    confort.cli(args=["train.long", "--steps", "7"], save_config=template)
    saved = tmp_path / "train.long" / "config.yaml"
    assert saved.exists()

    # subcommand name is read from the file; CLI flags still override it
    cfg = confort.cli(args=["--config-file", str(saved), "--lr", "0.1"], run=False)
    assert (cfg._entrypoint, cfg.steps, cfg.lr) == ("train.long", 7, 0.1)


def test_cli_single_entrypoint_needs_no_subcommand():
    @dataclass
    class Solo(Entrypoint):
        n: int = 1

        def run(self):
            return self.n

    Solo(_entrypoint="solo")
    assert confort.cli(args=["--n", "3"]) == 3
    assert confort.cli(args=["solo", "--n", "4"]) == 4


def test_cli_without_entrypoints_errors():
    with pytest.raises(RuntimeError, match="No entrypoints"):
        confort.cli(args=[])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_get_parser(project):
    parser = confort.get_parser()
    assert "train.fast" in parser.format_help()
