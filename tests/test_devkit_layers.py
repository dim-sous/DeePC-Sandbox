"""The shape of the devkit's packages and of every registered piece (BESS's
``tests/test_devkit_layers.py``, adapted).

The generic layers -- the records, the charts, the command line and the Lab -- are RunLab's; the
half of this file about their shape moved there with them (RunLab's ``tests/test_layers.py``),
and with RunLab's branch R1 so did the check that they name no piece, which RunLab holds
against its own made-up pieces. A piece is a package under ``harness/`` exposing exactly
``Config``, ``run`` and ``Record``, on the names RunLab holds for a project. Reading the
registry imports no piece.
"""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path

import param
import pytest

from runlab import Record, Settings
from harness import COMPONENTS, NAMES

ROOT = Path(__file__).resolve().parent.parent
DEVKIT = ROOT / "devkit"
#: Not a devkit package, but held like one: before the move the pieces could not import
#: ``figure``, ``terminal`` or ``lab``.
_RUNLAB = "runlab"
#: Who may import whom among the devkit's packages and RunLab, by dotted prefix. ``harness``
#: covers the registry and the pieces, which declare their settings and records on the names
#: RunLab holds for a project; how much of RunLab any file here may reach is held by
#: RunLab's own check (``tests/test_bench.py``). The bench's old simulation code, ``sim``, went
#: with the old closed loop; the pieces simulate through ``tools/system``.
_MAY_IMPORT = {"harness": {_RUNLAB}}


def _sources(package: str) -> list[Path]:
    return [p for p in (DEVKIT / package).rglob("*.py") if "__pycache__" not in p.parts]


def _imports(path: Path):
    """Every module a file imports by its full name; ``from a import b`` as ``a.b``, since ``b``
    may be a module (``from runlab import lab`` reaches the Lab)."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                yield f"{node.module}.{alias.name}"


def _within(name: str, prefixes: set[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def test_every_devkit_package_is_in_the_import_table() -> None:
    packages = {d.name for d in DEVKIT.iterdir() if (d / "__init__.py").exists()}
    assert packages == set(_MAY_IMPORT), f"devkit holds {sorted(packages)}; the table names {sorted(_MAY_IMPORT)}"


@pytest.mark.parametrize("package", sorted(_MAY_IMPORT))
def test_the_dependency_runs_one_way_between_the_layers(package: str) -> None:
    allowed = _MAY_IMPORT[package] | {package}
    held = set(_MAY_IMPORT) | {_RUNLAB}
    for path in _sources(package):
        crossing = sorted({n for n in _imports(path)
                           if n.split(".")[0] in held and not _within(n, allowed)})
        assert not crossing, (f"{path.relative_to(ROOT)} imports {crossing}; {package} may "
                              f"reach only {sorted(allowed)}")


@pytest.mark.parametrize("name", NAMES)
def test_every_registered_piece_is_the_template(name: str) -> None:
    piece = COMPONENTS[name]
    assert set(piece.__all__) >= {"Config", "run", "Record"}
    assert issubclass(piece.Config, Settings)
    assert callable(piece.run)
    assert issubclass(piece.Record, Record) and dataclasses.is_dataclass(piece.Record)
    assert piece.Record.component == name
    assert [f.name for f in dataclasses.fields(piece.Record)][:3] == ["name", "config", "provenance"]
    assert "numbers" in {f.name for f in dataclasses.fields(piece.Record)}
    piece.Config()                       # the defaults build


def _a_value(p: param.Parameter, default):
    """A value of ``p``'s type other than its default."""
    if isinstance(p, param.Boolean):
        return not default
    if isinstance(p, param.Selector):
        return next(o for o in p.objects if o != default)
    if isinstance(p, param.List):
        return [1.0]
    if isinstance(p, param.Integer):
        return 1 if default != 1 else 2
    return 1.0 if default != 1.0 else 2.0


def _settings(cls: type) -> dict[str, param.Parameter]:
    """A Config's settings as ``param`` holds them: every parameter but param's own ``name``."""
    return {n: p for n, p in cls.param.objects(instance=False).items() if n != "name"}


@pytest.mark.parametrize("name", NAMES)
def test_every_gate_a_config_declares_refuses_a_value_set_behind_it(name: str) -> None:
    cls = COMPONENTS[name].Config
    for setting, p in _settings(cls).items():
        if (p.metadata or {}).get("needs"):
            with pytest.raises(ValueError, match="applies only when"):
                cls(**{setting: _a_value(p, p.default)})


@pytest.mark.parametrize("name", NAMES)
def test_every_bound_a_config_declares_refuses_a_value_outside_it(name: str) -> None:
    cls = COMPONENTS[name].Config
    for setting, p in _settings(cls).items():
        bounds = (p.bounds if isinstance(p, param.Number) else None) or (None, None)
        for edge, outside in zip(bounds, (-1.0, 1.0)):
            if edge is not None:
                with pytest.raises(ValueError):
                    cls(**{setting: edge + outside})


def test_reading_the_registry_imports_no_piece() -> None:
    code = ("import sys, harness; "
            "assert not [m for m in sys.modules if m.startswith('harness.')], sorted(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
