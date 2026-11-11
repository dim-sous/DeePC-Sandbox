"""Three roots, one direction: src <- tools <- devkit, and src holds exactly what a controller runs.

`src/` is the DeePC controller a real system would run. It may not import the simulated world
(`tools/`) or the test bench (`devkit/`): on real hardware neither exists. `tools/` may import
`src/` and libraries, never the bench. The bench may import everything. (BESS's #1, adopted here
by the layout move; the three wrong-way imports it removed are in its PR.)

Stdlib only, so a broken package cannot break the test that explains why it is broken. The
forbidden package names are read off the tree, so a package added to tools/ or devkit/ tomorrow
is forbidden in src/ tomorrow, with no list to maintain. The one list that IS maintained is the
src inventory: adding a module to the controller is a reviewed act.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC, TOOLS, DEVKIT = ROOT / "src", ROOT / "tools", ROOT / "devkit"


def _packages(root: Path) -> list[str]:
    """The top-level packages a root holds, read off the tree."""
    return sorted(d.name for d in root.iterdir() if d.is_dir() and (d / "__init__.py").exists())


def _sources(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _importers(root: Path, packages: list[str]) -> list[str]:
    """The files under ``root`` whose code imports any of ``packages`` (flat names), at any
    indentation (a local import inside a function counts), or names one to ``import_module`` or
    ``__import__``."""
    names = "|".join(map(re.escape, packages))
    statement = re.compile(r"^\s*(from|import)\s+(%s)\b" % names, re.M)
    dynamic = re.compile(r"(import_module|__import__)\(\s*[\"'](%s)\b" % names)
    return sorted(str(p.relative_to(ROOT)) for p in _sources(root)
                  if statement.search(p.read_text()) or dynamic.search(p.read_text()))


def test_src_imports_neither_tools_nor_devkit() -> None:
    """The controller may not depend on the world it is simulated in, nor on the bench."""
    offenders = _importers(SRC, _packages(TOOLS) + _packages(DEVKIT))
    assert not offenders, (
        f"{offenders} import tools or devkit packages; the dependency runs ONE way — "
        f"src <- tools <- devkit, never the reverse"
    )


def test_tools_never_imports_devkit() -> None:
    """The world knows nothing about a run, a report or a tuner."""
    offenders = _importers(TOOLS, _packages(DEVKIT))
    assert not offenders, f"{offenders} import devkit packages; tools may import src and libraries only"


#: What the controller is. Two-way ratchet: an undeclared module fails, a declared one that
#: vanished fails. Decided 2026-11-11, the layout move: today's controller code, as it was.
#:   config            -> DeePCConfig (still holding bench settings: sim_duration, T_data,
#:                        noise_std_output; a later branch splits them out)
#:   constraints       -> the named bounds a plant hands the controller
#:   controller        -> the sparse QP
#:   hankel            -> block-Hankel matrices (build_data_matrices unused, D8)
#:   noise_estimator   -> the noise-adaptive variant's estimator (D16)
#:   online_hankel     -> the sliding window the controller builds its blocks in (D17)
#:   regularization    -> the persistency-of-excitation check (D13)
_SRC_INVENTORY = frozenset({
    "deepc/__init__.py",
    "deepc/config.py",
    "deepc/constraints.py",
    "deepc/controller.py",
    "deepc/hankel.py",
    "deepc/noise_estimator.py",
    "deepc/online_hankel.py",
    "deepc/regularization.py",
})


def test_src_holds_exactly_the_declared_inventory() -> None:
    actual = {str(p.relative_to(SRC)) for p in _sources(SRC)}
    undeclared, vanished = actual - _SRC_INVENTORY, _SRC_INVENTORY - actual
    assert not undeclared, (
        f"{sorted(undeclared)} appeared in src/ without being declared in the controller's "
        f"inventory — src holds only what a controller runs"
    )
    assert not vanished, f"{sorted(vanished)} are declared as the controller but do not exist"


def test_live_code_has_no_sys_path_hack() -> None:
    """The package installs; nothing edits sys.path to be importable."""
    pattern = re.compile(r"sys\.path\.(insert|append)")
    offenders = sorted(
        str(p.relative_to(ROOT)) for root in (SRC, TOOLS, DEVKIT) for p in _sources(root)
        if pattern.search(p.read_text())
    )
    assert not offenders, f"{offenders} edit sys.path; the editable install makes the roots importable"
