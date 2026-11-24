<p align="center">
  <img src="assets/logo.svg" alt="confort logo" width="200">
</p>

# confort

Dataclass configs, self-registering entrypoints and a [tyro](https://github.com/brentyi/tyro) CLI: a comfortable fort for your settings, reusable across projects.

```bash
pip install git+https://github.com/dbaieri/confort   # or: pip install -e . from a clone
```

Requires Python ≥ 3.10. Dependencies: `tyro`, `pyyaml`.

## In one minute

```python
# myproject/train.py
from dataclasses import dataclass
from confort import Entrypoint

@dataclass
class TrainerConfig(Entrypoint):
    """Train a model."""
    lr: float = 1e-3
    steps: int = 1000

@TrainerConfig.builds
class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
    def run(self):
        print(f"training for {self.config.steps} steps at lr={self.config.lr}")

# Creating a named instance registers it as a CLI subcommand.
TrainerConfig(lr=1e-4, _entrypoint="train.default")
TrainerConfig(steps=10, _entrypoint="train.debug", _help="Ten quick steps")
```

```python
# run.py
import confort

if __name__ == "__main__":
    confort.cli(discover="myproject")
```

```bash
python run.py --help
python run.py train.debug --lr 0.01
```

You don't need any `defaults.py` files or hand-written lists of commands. `discover` imports every module of the package, and each named entrypoint registers itself.

## Configs

All configs are regular `@dataclass` classes that subclass one of these bases.

| Class | What `setup(**kwargs)` does |
|---|---|
| `Config` | Nothing: a plain, printable, serializable config |
| `InstantiateConfig` | `target(self, **kwargs)`: builds the class this config describes |
| `FactoryConfig` | `getattr(_module, _name)(**public_fields, **kwargs)`: calls a library function or class directly |
| `Entrypoint` | Same as `InstantiateConfig`, plus CLI registration and `run()` |

**Private fields.** Fields whose name starts with `_` (`_target`, `_name`, `_module`, `_entrypoint`, ...) are hidden from the CLI, from `to_dict()` and from `str()`.

### `InstantiateConfig`: config → object

There are two ways to say which class a config builds:

```python
from dataclasses import dataclass, field
from typing import Type
from confort import InstantiateConfig

# 1. Decorator on the built class (no forward-reference lambda needed)
@dataclass
class ModelConfig(InstantiateConfig):
    width: int = 128

@ModelConfig.builds
class Model:
    def __init__(self, config: ModelConfig, device="cpu"): ...

# 2. A `_target` field default (the classic style)
@dataclass
class DataConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Data)
    root: str = "data/"

model = ModelConfig(width=256).setup(device="cuda")   # Model(config, device="cuda")
```

Subclasses inherit the target from `builds` unless they declare their own. If both are set, `_target` wins.

### `FactoryConfig`: config → library call

Use this to configure third-party callables without writing a wrapper class. `_module` can be a module or an import path, and `_name` can be dotted.

```python
import torch
from types import ModuleType
from confort import FactoryConfig

@dataclass
class AdamConfig(FactoryConfig):
    _module: ModuleType = torch.optim
    _name: str = "Adam"
    lr: float = 1e-3
    weight_decay: float = 0.0

optimizer = AdamConfig(lr=1e-4).setup(params=model.parameters())
```

### `default(value)`: mutable defaults

`default(value)` is a shorthand for `field(default_factory=lambda: copy.deepcopy(value))`. It works for lists, dicts and nested configs:

```python
from confort import default

@dataclass
class TrainerConfig(Entrypoint):
    milestones: list[int] = default([100, 200])
    optimizer: FactoryConfig = default(AdamConfig(lr=1e-4))
```

A field annotated with a base class (`FactoryConfig`) can hold any subclass. On the CLI, it exposes the fields of the subclass you set as its default.

### Common `Config` methods

| Method | Description |
|---|---|
| `str(cfg)` | Indented, human-readable view (public fields only) |
| `cfg.to_dict(include_private=False)` | Nested plain data (dicts, lists, scalars), e.g. for experiment loggers |
| `cfg.replace(**overrides)` | Copy with changes; dotted keys reach nested configs: `cfg.replace(**{"optimizer.lr": 1e-5})` |
| `cfg.save(path)` | Write YAML (`.yaml`/`.yml`) or JSON (`.json`), creating parent directories |
| `cfg.merge(path_or_dict)` | Copy with values from a file or dict. Nested configs keep their class, and unknown keys raise `KeyError` |

## Entrypoints

An `Entrypoint` is an `InstantiateConfig` that can be launched from the CLI:

- **Registration.** Pass `_entrypoint="name"` (and optionally `_help="..."`) when you create an instance, and it registers itself when its module is imported. The instance's values become the CLI defaults. Instances without a name are ordinary configs.
- **Running.** `cfg.run()` calls `cfg.setup().run()`. To change this, override `run` in your subclass.
- **Many commands, one class.** Create as many named instances of the same class as you like:
  ```python
  seg_dense  = SegConfig(support="dataset", _entrypoint="seg.dense")
  seg_sparse = SegConfig(support="sparse",  _entrypoint="seg.sparse")
  ```
- **Variants.** `cfg.replace(...)` returns an unregistered copy. Pass a new `_entrypoint` to register the copy as another command:
  ```python
  seg_sparse.replace(**{"knn": 20}, _entrypoint="seg.sparse-k20")
  ```
- **Name clashes.** Registering a different config under an existing name raises `ValueError`. Re-registering an identical config (for example, a module imported twice) is ignored.

### Function entrypoints

Small tools don't need a config class. Decorate a function and its signature becomes the CLI:

```python
import confort

@confort.command("data.stats", help="Print dataset statistics")
def stats(root: str, verbose: bool = False) -> None:
    ...
```

Without a name, the function name is used with `_` replaced by `-`. If you leave out `help`, the docstring is used instead.

## The CLI

```python
confort.cli(
    discover="myproject",                          # package(s) to import recursively
    exclude=["myproject.viz"],                     # skip modules/packages (glob patterns ok)
    save_config="outputs/{name}/{time}/config.yaml",
    accent_color="bright_yellow",
)
```

| Argument | Default | Description |
|---|---|---|
| `discover` | `None` | Package name or list of names to import recursively. `__main__` modules are always skipped. Import errors are raised, not hidden |
| `exclude` | `()` | Module names or glob patterns to skip. Excluding a package skips its subpackages |
| `args` | `sys.argv[1:]` | Arguments to parse |
| `prog`, `description` | `None` | Shown in `--help` |
| `run` | `True` | Run the chosen entrypoint and return its result. With `False`, return the parsed config instead |
| `save_config` | `None` | Save the parsed config before running. `{name}` and `{time}` are filled in |
| `config_file_flag` | `"--config-file"` | Flag that loads a saved YAML/JSON config. `None` disables it |
| `accent_color` | `None` | tyro help color |
| `markers` | `confort.DEFAULT_MARKERS` | tyro markers. By default, fixed fields are hidden and booleans are written `--flag True` / `--flag False` (`tyro.conf.FlagConversionOff`) |
| `registry` | `confort.registry` | Registry to read entrypoints from |

Command-line behavior:

```bash
python run.py train.default --lr 3e-4 --optimizer.weight-decay 0.01   # nested fields use dots
python run.py --config-file outputs/train.default/2026-11-24_10-00-00/config.yaml --steps 50
```

- **Loading a saved config.** `--config-file` loads values from a saved file, and any other flags override them. Files written by `cfg.save` / `save_config` record their entrypoint name, so you don't need to repeat the subcommand.
- **Single entrypoint.** With only one entrypoint, the CLI has no subcommand: run `python run.py --lr ...` directly. Its name is still accepted.
- **Choose a non-clashing flag name.** If one of your configs has a field called `config_file`, set `config_file_flag` to another flag name.
- **pyproject scripts.** You can use `confort.cli` from a `[project.scripts]` function:
  ```python
  def main():
      confort.cli(discover="myproject")
  ```

### Using the CLI type with tyro directly

`confort.cli_type()` returns the tyro type that `confort.cli` parses: a `Union` with one subcommand per registered entrypoint, or just that subcommand's type if there is only one. Use it when you want to drive tyro yourself, for example to build your own launcher or to feed tyro-based tooling.

Always pass `registry=confort.tyro_registry` along with it. That constructor registry hides private fields from the CLI and stops parsed copies from re-registering. Without it, overriding a value of a named entrypoint raises a name-clash error.

```python
import tyro
import confort

config = tyro.cli(
    confort.cli_type(discover="myproject"),   # same discover/exclude/registry options as cli()
    registry=confort.tyro_registry,
    config=confort.DEFAULT_MARKERS,
)
config.run()   # cli_type leaves running, saving and --config-file to you
```

## Registry and discovery

| Function / object | Description |
|---|---|
| `confort.discover(packages, exclude=())` | Import packages recursively. Returns the names of the imported modules |
| `confort.registry` | Global `Registry`: `names()`, `registry["name"]`, `in`, `len`, iteration, `remove(name)`, `clear()` |
| `confort.Registration` | One entry, with fields `name`, `target`, `help` and `module` |
| `confort.suppress_registration()` | Context manager: entrypoints created inside it are not registered |
| `confort.cli_type(discover=None, exclude=(), registry=None)` | The tyro type behind `confort.cli` (see above) |
| `confort.tyro_registry` | tyro `ConstructorRegistry` to pass as `registry=` whenever you give confort configs to tyro yourself |

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests in `tests/` use a small example package, `tests/fakeproj`, that shows the intended project layout.

## License

MIT
