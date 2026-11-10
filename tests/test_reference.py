"""The safety net: today's code against its frozen numbers, bit for bit.

A branch that means to move no number (a move, a rename, a new layer around the code) proves
it by keeping these green; one that means to move them remakes the files
(``uv run python -m tests.reference.make``) and says so in its PR. The rule is BESS's, written
after its incident B4 (a proof that compared nothing): a test that checks a key exists proves
nothing, so every array is compared by its bytes.

What each case runs, and the conditions, are in ``tests/reference/cases.py`` and, as made, in
``tests/reference/manifest.json``. Bit-identity was measured on one computer only; a mismatch
names the library versions the files were made with beside today's.
"""

from __future__ import annotations

import json
import math
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pytest

from tests.reference import cases

MANIFEST = json.loads(cases.MANIFEST.read_text())


def _installed(lib: str) -> str:
    try:
        return version(lib)
    except PackageNotFoundError:
        return "not installed"


def _versions() -> str:
    made = MANIFEST["made"]["versions"]
    return ", ".join(f"{lib} {made[lib]} (now {_installed(lib)})" for lib in made)


def _same(a, b) -> bool:
    """Equal, a NaN equal to a NaN: a metrics file holds NaN where a metric had no answer."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _difference(frozen: np.ndarray, now: np.ndarray) -> str:
    """Where two arrays first differ, in words."""
    if frozen.shape != now.shape or frozen.dtype != now.dtype:
        return f"shape/dtype {frozen.shape} {frozen.dtype} -> {now.shape} {now.dtype}"
    a = np.ascontiguousarray(frozen).reshape(-1).view(np.uint8).reshape(frozen.size, -1)
    b = np.ascontiguousarray(now).reshape(-1).view(np.uint8).reshape(now.size, -1)
    unequal = np.flatnonzero((a != b).any(axis=1))        # entries whose bytes differ
    first = np.unravel_index(int(unequal[0]), frozen.shape)
    words = (f"{unequal.size} of {frozen.size} entries differ, first at {tuple(map(int, first))}: "
             f"{frozen[first]!r} -> {now[first]!r}")
    if np.issubdtype(frozen.dtype, np.floating):
        words += f"; largest |change| {float(np.nanmax(np.abs(now - frozen))):.3g}"
    return words


def test_every_frozen_file_is_a_case_and_every_case_is_frozen() -> None:
    """A case taken out of the registry would otherwise drop out of the net with its file left
    behind, and the suite stay green."""
    npz, run_py = cases.frozen_names()
    assert npz == set(cases.CASES), (f"files with no case: {sorted(npz - set(cases.CASES))}; "
                                     f"cases with no file: {sorted(set(cases.CASES) - npz)}")
    assert run_py == set(cases.RUN_PY), (f"files with no case: {sorted(run_py - set(cases.RUN_PY))}"
                                         f"; cases with no file: {sorted(set(cases.RUN_PY) - run_py)}")
    assert set(MANIFEST["cases"]) == set(cases.CASES)
    assert set(MANIFEST["run_py"]) == set(cases.RUN_PY)


@pytest.mark.parametrize("name", cases.CASES)
def test_the_case_runs_under_its_frozen_conditions(name: str) -> None:
    """Every setting the case ran under, as made: a default changed under a case is a change
    to its conditions even where no number moves."""
    assert name in MANIFEST["cases"], (
        f"{name} has no frozen conditions; make them: uv run python -m tests.reference.make")
    assert cases.conditions_of(name) == MANIFEST["cases"][name]


@pytest.mark.parametrize("name", cases.CASES)
def test_the_case_produces_its_frozen_arrays(name: str) -> None:
    path = cases.REFERENCE_DIR / f"{name}.npz"
    assert path.exists(), f"{name} has no frozen file; make it: uv run python -m tests.reference.make"
    with np.load(path) as stored:
        frozen = {key: stored[key] for key in stored.files}
    now = cases.produce(name)
    assert sorted(now) == sorted(frozen), f"{name}: arrays {sorted(frozen)} -> {sorted(now)}"
    changed = {key: _difference(frozen[key], now[key]) for key in frozen
               if frozen[key].dtype != now[key].dtype or frozen[key].shape != now[key].shape
               or frozen[key].tobytes() != now[key].tobytes()}
    assert not changed, (f"{name} moved: " + "; ".join(f"{k}: {w}" for k, w in changed.items())
                         + f". Made with {_versions()}")


@pytest.mark.parametrize("name", list(cases.RUN_PY))
def test_run_py_runs_under_its_frozen_conditions(name: str) -> None:
    """Every option ``run.py`` parses, its defaults included, and every config field it builds."""
    frozen = {k: v for k, v in MANIFEST["run_py"][name].items() if k != "dropped"}
    assert cases.run_py_conditions(name) == frozen


@pytest.mark.parametrize("name", list(cases.RUN_PY))
def test_run_py_writes_its_report_and_its_frozen_metrics(name: str, tmp_path) -> None:
    """``run.py`` end to end: a report and a metrics file are written, and every metric but
    the wall-clock ones equals the frozen one."""
    frozen = json.loads((cases.REFERENCE_DIR / f"{name}.json").read_text())
    now = cases.run_py_metrics(name, tmp_path)
    changed = {k: (frozen.get(k), now.get(k)) for k in set(frozen) | set(now)
               if not _same(frozen.get(k), now.get(k))}
    assert not changed, f"{name} metrics moved (frozen, now): {changed}. Made with {_versions()}"
