"""The shape of the devkit's packages and of every registered piece (BESS's
``tests/test_devkit_layers.py``, adapted).

The generic layers -- the records, the charts, the command line and the Lab -- are Benchlab's; the
half of this file about their shape moved there with them (Benchlab's ``tests/test_layers.py``).
That they never name one of DeePC's pieces stays here, read off the Benchlab DeePC installs: only
this repository knows its pieces' names. A piece is a package under ``harness/`` exposing exactly
``Config``, ``run`` and ``Record``, and reaches Benchlab's record layer alone. Reading the registry
imports no piece.
"""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path

import param
import pytest

import benchlab
from benchlab import record
from harness import COMPONENTS, NAMES
from benchlab.record.config import Settings, fields_of

ROOT = Path(__file__).resolve().parent.parent
DEVKIT = ROOT / "devkit"
#: Not a devkit package, but held like one: before the move ``sim`` could not import ``record``,
#: and the pieces could not import ``figure``, ``terminal`` or ``lab``.
_BENCHLAB = "benchlab"
#: Who may import whom among the devkit's packages and Benchlab, by dotted prefix. ``harness``
#: covers the registry and the pieces, which declare their settings and records on Benchlab's
#: record layer and reach nothing else of it; ``sim`` is the bench's simulation code the pieces
#: drive, and reaches neither.
_MAY_IMPORT = {"sim": set(), "harness": {"sim", f"{_BENCHLAB}.record"}}
#: The Benchlab DeePC runs on, as installed: the editable working copy or the pinned tag alike.
BENCHLAB_SOURCES = sorted(p for p in Path(benchlab.__file__).parent.rglob("*.py")
                          if "__pycache__" not in p.parts)


def _sources(package: str) -> list[Path]:
    return [p for p in (DEVKIT / package).rglob("*.py") if "__pycache__" not in p.parts]


def _imports(path: Path):
    """Every module a file imports by its full name; ``from a import b`` as ``a.b``, since ``b``
    may be a module (``from benchlab import lab`` reaches the Lab)."""
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
    held = set(_MAY_IMPORT) | {_BENCHLAB}
    for path in _sources(package):
        crossing = sorted({n for n in _imports(path)
                           if n.split(".")[0] in held and not _within(n, allowed)})
        assert not crossing, (f"{path.relative_to(ROOT)} imports {crossing}; {package} may "
                              f"reach only {sorted(allowed)}")


@pytest.mark.parametrize("path", BENCHLAB_SOURCES,
                         ids=[str(p.relative_to(Path(benchlab.__file__).parent)) for p in BENCHLAB_SOURCES])
def test_benchlab_never_names_one_of_these_pieces_in_its_code(path: Path) -> None:
    """A string literal, an attribute or an imported name equal to a piece's name couples the
    bench to that piece; it was this file's check on the layers before they moved (Benchlab holds
    it against its own made-up pieces only). Prose, comments and local variables cannot couple."""
    names = set(NAMES)
    for node in ast.walk(ast.parse(path.read_text())):
        spelled = ((isinstance(node, ast.Constant) and node.value in names)
                   or (isinstance(node, ast.Attribute) and node.attr in names)
                   or (isinstance(node, ast.alias) and node.name in names))
        assert not spelled, f"benchlab/{path.relative_to(Path(benchlab.__file__).parent)}:{node.lineno} names a piece"


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
