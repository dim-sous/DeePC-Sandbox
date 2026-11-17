"""The safety net's cases: what each piece produces at stated settings, frozen so that a change
meant to move no number can show it moved none (CLAUDE.md, "a refactor proves itself"; BESS
[B4]). Every case runs a piece's own ``run(Config)``; a piece that reads a saved run (Open loop
reads a System run, Closed loop an Open loop run) gets its upstream case's record, saved once per
process in a temporary folder.

What a case keeps (``produce``): every array field of its record, every number (``numbers.<key>``)
and every constant (``constants.<key>``, nested keys joined by dots), each as a NumPy array; the
Closed loop's statuses and solvers as string arrays. Left out (``LEFT_OUT``): wall-clock times,
provenance, and the upstream file's path, date and commit, which differ between two runs of the
same code. The conditions (``conditions_of``) are every setting of the case's Config, the
temporary path of an upstream run replaced by the upstream case's name.

The comparison is the developer's (2026-11-17, "Bit for bit, per computer"): bit for bit on the
computer that made the files (``computer()``, recorded in the manifest), within a stated
tolerance on another (``tests/test_reference.py``).
"""

from __future__ import annotations

import atexit
import functools
import platform
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from harness import LIBRARIES, get_component

ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE_DIR = Path(__file__).resolve().parent
MANIFEST = REFERENCE_DIR / "manifest.json"
#: The code the numbers come from, fingerprinted in the manifest.
CODE = ("src", "tools", "devkit")

#: name -> (piece, upstream case or None, the settings off the piece's defaults).
CASES: dict[str, tuple[str, str | None, dict]] = {
    "system-default": ("system", None, {}),
    "system-pendulum": ("system", None, {"f": ["x2", "-sin(x1) - theta1*x2 + u1"],
                                         "h": ["x1"], "theta": [0.2]}),
    "open_loop-default": ("open_loop", "system-default", {}),
    "open_loop-clipped": ("open_loop", "system-default",
                          {"u_exc_min": [-6.0, -6.0], "u_exc_max": [6.0, 6.0], "sigma": 0.0}),
    "closed_loop-default": ("closed_loop", "open_loop-default", {"duration": 2.0}),
    "closed_loop-step": ("closed_loop", "open_loop-default",
                         {"r": ["Heaviside(t - 1)", "0", "-0.5*Heaviside(t - 1)", "0"],
                          "duration": 1.0}),
}
#: The setting through which a piece names its upstream run.
UPSTREAM_SETTING = {"open_loop": "system_run", "closed_loop": "open_loop_run"}
#: What a case does not keep, per piece: wall-clock values and what names a file rather than
#: what it holds.
LEFT_OUT = {
    "system": [],
    "open_loop": ["numbers.step_time_mean", "numbers.step_time_std", "numbers.step_time_max",
                  "constants.system_run"],
    "closed_loop": ["solve_time", "numbers.solve_time_mean", "numbers.solve_time_std",
                    "numbers.solve_time_max", "constants.open_loop_run"],
}


def computer() -> dict[str, str]:
    """What the numbers' last bits depend on: the OS, the machine, the CPU model, Python and the
    versions of the libraries a record names (``harness.LIBRARIES``)."""
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    def installed(lib: str) -> str:
        try:
            return version(lib)
        except PackageNotFoundError:
            return "not installed"
    return {"system": platform.system(), "machine": platform.machine(), "cpu": cpu,
            "python": platform.python_version(),
            **{lib: installed(lib) for lib in LIBRARIES}}


@functools.cache
def _folder() -> Path:
    folder = Path(tempfile.mkdtemp(prefix="deepc-reference-"))
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    return folder


@functools.cache
def record_of(name: str):
    """The case's record, run once per process; saved in the temporary folder, where a
    downstream case reads it."""
    piece, upstream, settings = CASES[name]
    component = get_component(piece)
    extra = {}
    if upstream is not None:
        record_of(upstream)
        extra[UPSTREAM_SETTING[piece]] = str(_folder() / f"{upstream}.json")
    record = component.run(component.Config(**settings, **extra))
    record.save(_folder() / f"{name}.json")
    return record


def _flat(prefix: str, value, out: dict[str, np.ndarray]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            _flat(f"{prefix}.{key}", inner, out)
    else:
        out[prefix] = np.asarray(value)


def produce(name: str) -> dict[str, np.ndarray]:
    """What the case keeps, by key (see the module)."""
    piece = CASES[name][0]
    record = record_of(name)
    out: dict[str, np.ndarray] = {}
    skip = {"name", "config", "provenance", "constants", "numbers"}
    for field in record.__dataclass_fields__:
        if field not in skip:
            out[field] = np.asarray(getattr(record, field))
    _flat("numbers", record.numbers, out)
    _flat("constants", record.constants, out)
    for key in LEFT_OUT[piece]:
        for k in [k for k in out if k == key or k.startswith(key + ".")]:
            del out[k]
    pickled = sorted(k for k, v in out.items() if v.dtype == object)
    if pickled:      # np.savez would pickle them and the test's np.load refuse them
        raise ValueError(f"{name}: {pickled} are not plain arrays (None or ragged values); a "
                         f"frozen value is a number or a string")
    return out


def conditions_of(name: str) -> dict:
    """Every setting the case ran under, the upstream run named by its case."""
    piece, upstream, _settings = CASES[name]
    cfg = record_of(name).config
    out = {k: getattr(cfg, k) for k in cfg.param if k != "name"}
    if upstream is not None:
        out[UPSTREAM_SETTING[piece]] = f"case {upstream}"
    return {"piece": piece, "settings": out}
