"""The shape of the devkit's layers and of every registered piece (BESS's
``tests/test_devkit_layers.py``, adapted).

The generic layers -- ``record`` (settings + records) and ``terminal`` (the command) -- only ever
touch a Config or a Record and never know what is inside: if a line of code in them names a
piece, it is in the wrong layer. A piece is a package under ``harness/`` exposing exactly
``Config``, ``run`` and ``Record``. Reading the registry imports no piece.
"""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path

import param
import pytest

import record
from harness import COMPONENTS, NAMES
from record.config import Settings, fields_of

ROOT = Path(__file__).resolve().parent.parent
DEVKIT = ROOT / "devkit"
GENERIC = ("record", "terminal")
#: Who may import whom among the devkit's packages. ``harness`` covers the registry and the
#: pieces; ``sim`` is the bench's simulation code the pieces drive.
_MAY_IMPORT = {"record": set(), "sim": set(), "harness": {"record", "sim"},
               "terminal": {"record", "harness"}}


def _sources(package: str) -> list[Path]:
    return [p for p in (DEVKIT / package).rglob("*.py") if "__pycache__" not in p.parts]


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def test_every_devkit_package_is_in_the_import_table() -> None:
    packages = {d.name for d in DEVKIT.iterdir() if (d / "__init__.py").exists()}
    assert packages == set(_MAY_IMPORT), f"devkit holds {sorted(packages)}; the table names {sorted(_MAY_IMPORT)}"


@pytest.mark.parametrize("package", sorted(_MAY_IMPORT))
def test_the_dependency_runs_one_way_between_the_layers(package: str) -> None:
    allowed = _MAY_IMPORT[package] | {package}
    others = set(_MAY_IMPORT) - allowed
    for path in _sources(package):
        crossing = sorted({n for n in _imports(path) if n.split(".")[0] in others})
        assert not crossing, (f"{path.relative_to(ROOT)} imports {crossing}; {package} may "
                              f"reach only {sorted(allowed)}")


@pytest.mark.parametrize("package", GENERIC)
def test_a_generic_layer_imports_the_registry_and_never_a_piece(package: str) -> None:
    """``harness`` is the registry, how the terminal finds a piece; ``harness.<x>`` is a piece."""
    for path in _sources(package):
        deeper = sorted({n for n in _imports(path) if n.startswith("harness.")})
        assert not deeper, f"{path.relative_to(ROOT)} imports a piece: {deeper}"


@pytest.mark.parametrize("package", GENERIC)
def test_a_generic_layer_never_names_a_piece_in_its_code(package: str) -> None:
    """A string literal, an attribute or an imported name equal to a piece's name couples a
    generic layer to that piece. Prose, comments and local variables cannot couple."""
    names = set(NAMES)
    for path in _sources(package):
        for node in ast.walk(ast.parse(path.read_text())):
            spelled = ((isinstance(node, ast.Constant) and node.value in names)
                       or (isinstance(node, ast.Attribute) and node.attr in names)
                       or (isinstance(node, ast.alias) and node.name in names))
            assert not spelled, f"{path.relative_to(ROOT)}:{node.lineno} names a piece"


@pytest.mark.parametrize("name", NAMES)
def test_every_registered_piece_is_the_template(name: str) -> None:
    piece = COMPONENTS[name]
    assert set(piece.__all__) >= {"Config", "run", "Record"}
    assert issubclass(piece.Config, Settings)
    assert callable(piece.run)
    assert issubclass(piece.Record, record.Record) and dataclasses.is_dataclass(piece.Record)
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


@pytest.mark.parametrize("name", NAMES)
def test_every_gate_a_config_declares_refuses_a_value_set_behind_it(name: str) -> None:
    cls = COMPONENTS[name].Config
    params = cls.param.objects(instance=False)
    gated = [f for f in fields_of(cls) if f["needs"]]
    for f in gated:
        with pytest.raises(ValueError, match="applies only when"):
            cls(**{f["name"]: _a_value(params[f["name"]], f["default"])})


@pytest.mark.parametrize("name", NAMES)
def test_every_bound_a_config_declares_refuses_a_value_outside_it(name: str) -> None:
    cls = COMPONENTS[name].Config
    for f in fields_of(cls):
        for edge, outside in ((f["low"], -1.0), (f["high"], 1.0)):
            if edge is not None:
                with pytest.raises(ValueError):
                    cls(**{f["name"]: edge + outside})


def test_reading_the_registry_imports_no_piece() -> None:
    code = ("import sys, harness; "
            "assert not [m for m in sys.modules if m.startswith('harness.')], sorted(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
