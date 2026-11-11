"""The record layer: what one run of a piece leaves behind, as one JSON file. Adapted from BESS's
``devkit/record/__init__.py`` (itself RCWS's), trimmed to what this repository's pieces use.

A piece's ``Record`` is a frozen dataclass whose first field is ``name`` (also the file stem) and
which carries ``config`` (the Config the run used), ``provenance`` and ``numbers: dict[str,
float]``, plus the arrays its figures will draw. It declares, from its own fields, the panels a
figure layer may draw (``series()``) and the words and units of its numbers (``number_specs()``),
so a drawing and a listing cannot drift from the data.

A number with no answer is left out of ``numbers``, never saved as NaN. Non-finite values inside
an array ride as the tokens ``"nan"``, ``"inf"``, ``"-inf"`` and load back as floats. A float is
written with Python's shortest exact repr, so an array loads back bit for bit.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import types
import typing
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from record.config import Settings, from_dict, to_dict

__all__ = ["Provenance", "stamp", "Series", "bound_refs", "Number", "Record", "read_header", "capital", "REPO_ROOT",
           "COMPONENT_KEY", "PANELS_KEY", "NUMBER_SPECS_KEY", "PANEL_GROUPS_KEY", "HEADER_KEYS"]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

COMPONENT_KEY, PANELS_KEY, NUMBER_SPECS_KEY = "component", "panels", "number_specs"
#: Each panel's group, ``{panel: group}``.
PANEL_GROUPS_KEY = "panel_groups"
HEADER_KEYS = (COMPONENT_KEY, PANELS_KEY, NUMBER_SPECS_KEY, PANEL_GROUPS_KEY)
_NONFINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
#: The libraries a run's numbers depend on, recorded with every run.
_LIBRARIES = ("numpy", "scipy", "cvxpy", "osqp", "scs")


# ---------------------------------------------------------------------------
#  Provenance
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Provenance:
    """What produced a record. Stamped by ``stamp``; never hand-written."""

    git_sha: str
    git_dirty: bool
    date: str              # UTC, ISO 8601
    duration_s: float
    versions: dict         # {library: version}, _LIBRARIES


def _git(*args: str) -> str:
    # check=True: a record stamped from nowhere would read as a record from nowhere.
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def stamp(duration_s: float) -> Provenance:
    return Provenance(git_sha=_git("rev-parse", "--short", "HEAD"),
                      git_dirty=bool(_git("status", "--porcelain")),
                      date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      duration_s=float(duration_s),
                      versions={lib: version(lib) for lib in _LIBRARIES})


# ---------------------------------------------------------------------------
#  Declarations a figure layer and a listing read
# ---------------------------------------------------------------------------

def capital(words: str) -> str:
    """``words`` with a capital first letter and the rest as written: how a label, a legend and a
    heading begin (the developer's rule, BESS 2026-10-27), acronyms and symbols keeping their own."""
    return words[:1].upper() + words[1:]


@dataclasses.dataclass(frozen=True)
class Series:
    """One panel: an abscissa, one or more traces on one ordinate unit, optional references and
    bands. The unit lives in the axis label -- one unit per panel."""

    name: str                                        # the figure's file stem, kebab-case
    x: Any                                           # (M,)
    x_label: str                                     # with its unit
    y_label: str                                     # with its unit
    traces: tuple[tuple[str, Any], ...]              # (legend label, (M,) values)
    refs: tuple[tuple[str, float], ...] = ()         # (label, constant value)
    bands: tuple[tuple[str, Any, Any], ...] = ()     # (label, lo, hi): scalars or (M,) arrays
    log_y: bool = False
    doc: str = ""
    group: str = ""                                  # the heading a listing puts it under
    #: The words a listing and a card show for the panel; left out, its name. The name stays the
    #: file's stem and the address's key, kebab-case, whatever the words.
    title: str = ""
    #: The statistics its traces may be pooled by across runs ("mean", "rms"); () for none.
    pooled: tuple[str, ...] = ()
    #: Each trace's own abscissa, in ``traces`` order, for a panel whose traces do not share one
    #: (a path: y against x, the plant's and the reference's); () when they share ``x``.
    trace_x: tuple[Any, ...] = ()
    #: One unit of the bottom axis drawn as long as one unit of the ordinate (a path in metres).
    equal_axes: bool = False

    def __post_init__(self) -> None:
        if not self.title:
            object.__setattr__(self, "title", self.name)
        if not self.traces:
            raise ValueError(f"panel {self.name!r} declares no trace")
        unknown = sorted(set(self.pooled) - {"mean", "rms"})
        if unknown:
            raise ValueError(f"panel {self.name!r} pools by {unknown}; a panel pools by "
                             f"'mean' or 'rms'")
        if self.trace_x:
            if len(self.trace_x) != len(self.traces):
                raise ValueError(f"panel {self.name!r}: {len(self.trace_x)} abscissae for "
                                 f"{len(self.traces)} traces")
            if self.bands or self.pooled:
                raise ValueError(f"panel {self.name!r}: a band or a pooling needs one abscissa "
                                 f"the traces share; its traces each have their own")
            for (label, y), xs in zip(self.traces, self.trace_x):
                if len(y) != len(xs):
                    raise ValueError(f"panel {self.name!r}: trace {label!r} has {len(y)} values "
                                     f"against {len(xs)} abscissa values")
            return
        n = len(self.x)
        for label, y in self.traces:
            if len(y) != n:
                raise ValueError(f"panel {self.name!r}: trace {label!r} has {len(y)} values "
                                 f"against {n} abscissa values")
        for label, lo, hi in self.bands:
            for edge in (lo, hi):
                if np.ndim(edge) and len(edge) != n:
                    raise ValueError(f"panel {self.name!r}: band {label!r} edge has "
                                     f"{len(edge)} values against {n}")


def bound_refs(bounds: Any, quantity: str, unit: str) -> tuple[tuple[str, float], ...]:
    """A panel's reference lines for ``quantity``'s ``[lower, upper]`` bounds in ``unit``, each a
    finite one (an infinite bound is no line), named with the quantity and its unit (CLAUDE.md:
    "A bound names the quantity it bounds, with its unit", D7)."""
    return tuple((f"{side} bound of {quantity} [{unit}]", float(v))
                 for side, v in zip(("Lower", "Upper"), np.asarray(bounds, dtype=float))
                 if np.isfinite(v))


@dataclasses.dataclass(frozen=True)
class Number:
    """The words and unit of one summary number in ``Record.numbers``."""

    name: str
    label: str
    unit: str = ""          # "" for a dimensionless quantity
    headline: bool = False  # shown first where a listing folds its columns
    doc: str = ""
    group: str = ""         # the heading a listing puts it under


# ---------------------------------------------------------------------------
#  JSON encoding
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, Settings):
        return to_dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else ("nan" if math.isnan(f) else ("inf" if f > 0 else "-inf"))
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _array_back(value: Any) -> np.ndarray:
    def back(v: Any) -> Any:
        if isinstance(v, list):
            return [back(x) for x in v]
        if isinstance(v, str):
            if v not in _NONFINITE:                  # a ValueError, so `load` refuses it in words
                raise ValueError(f"{v!r} is neither a number nor one of {sorted(_NONFINITE)}")
            return _NONFINITE[v]
        return v
    return np.asarray(back(value), dtype=float)


def _decode(hint: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        return _decode(args[0], value) if len(args) == 1 else value
    if isinstance(hint, type) and issubclass(hint, Settings):
        return from_dict(hint, value, where="config")
    if hint is Provenance:
        return Provenance(**value)
    if hint is np.ndarray:
        return _array_back(value)
    return value


# ---------------------------------------------------------------------------
#  Record
# ---------------------------------------------------------------------------

class Record:
    """Base of every piece's Record; see the module. Subclasses are frozen dataclasses."""

    component: ClassVar[str] = ""        # the registered piece name (the command word)
    label: ClassVar[str] = ""            # the piece's own words
    produced_by: ClassVar[str] = ""      # the command that writes one

    def series(self) -> tuple[Series, ...]:
        return ()

    def number_specs(self) -> tuple[Number, ...]:
        return ()

    # -- the header a listing reads without rebuilding the body
    def header(self) -> dict[str, Any]:
        series = self.series()
        return {COMPONENT_KEY: type(self).component,
                PANELS_KEY: [s.name for s in series],
                NUMBER_SPECS_KEY: [dataclasses.asdict(n) for n in self.number_specs()],
                PANEL_GROUPS_KEY: {s.name: s.group for s in series}}

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write header then fields, atomically. Refuses a non-finite or undeclared number."""
        numbers = getattr(self, "numbers", {})
        bad = sorted(k for k, v in numbers.items() if not math.isfinite(float(v)))
        if bad:
            raise ValueError(f"{type(self).component} numbers {bad} are not finite; a number "
                             f"with no answer is left out of the dict, not saved as a token")
        undeclared = sorted(set(numbers) - {n.name for n in self.number_specs()})
        if undeclared:
            raise ValueError(f"{type(self).component} numbers {undeclared} carry no words "
                             f"(number_specs); every number a listing shows is declared")
        payload = dict(self.header())
        payload.update((f.name, _jsonable(getattr(self, f.name)))
                       for f in dataclasses.fields(self))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".partial")
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=1, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Record":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no {cls.__name__} record at {path}. Run it first:\n"
                                    f"    {cls.produced_by}\nwhich writes that file.")
        with open(path) as fh:
            raw = json.load(fh)
        component = raw.get(COMPONENT_KEY)
        if component != cls.component:
            raise ValueError(f"{path} is a {component!r} record, not {cls.component!r}")
        body = {k: v for k, v in raw.items() if k not in HEADER_KEYS}
        hints = typing.get_type_hints(cls)
        names = [f.name for f in dataclasses.fields(cls)]
        unknown, missing = sorted(set(body) - set(names)), sorted(set(names) - set(body))
        if unknown or missing:
            raise ValueError(f"the record at {path} does not match the current {cls.__name__} "
                             f"(unknown fields {unknown}, missing {missing}); it predates or "
                             f"postdates a field. Re-run: {cls.produced_by}")
        try:
            return cls(**{n: _decode(hints[n], body[n]) for n in names})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"the record at {path} does not load as the current "
                             f"{cls.__name__} ({exc}). Re-run: {cls.produced_by}") from exc


def read_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The header and the name of a record file, without importing its piece."""
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not a record: not JSON ({exc.msg} at line {exc.lineno})") from None
    if not isinstance(raw, dict) or COMPONENT_KEY not in raw:
        raise ValueError(f"{path} is not a record: it carries no {COMPONENT_KEY!r} header")
    return {k: raw[k] for k in (*HEADER_KEYS, "name") if k in raw}
