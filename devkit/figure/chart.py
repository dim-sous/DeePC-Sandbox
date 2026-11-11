"""What goes on a panel, decided once, for a still and the Lab's page alike.

Two kinds of figure over ``pairs`` -- ``(run name, record)`` in the order the runs are shown:

* :func:`chart_of`, a panel the records declare (``Record.series``): one run alone; several on
  one pair of axes (``runs="one"``: the colour carries the run or the quantity, the other told
  apart by a shade of that hue and a dash, and a trace or band every run holds identically is
  drawn once and said to be); a row each (``"rows"``, on one time axis); or the runs pooled
  into one trace per quantity (``"mean"``, ``"rms"``) where the panel declares that statistic;
* :func:`sweep_of`, one summary number against a setting: every run a point, coloured by the
  settings that tell the runs apart but a replicate (the seed), runs of one colour at one value
  side by side in replicate order, or pooled into their mean or RMS there.

Every keyword is a way of shaping the drawing; what the figure cannot draw it refuses in words
(:func:`pool_refusal`, :func:`render.log_refusal`), which a view shows where it greys a control.
Nothing here names a piece: it reads the declarations a record carries. BESS's
``devkit/figure/chart.py``, without its chi-square bands (DeePC's records declare none), and with
two refusals of its own: runs of different pieces, and a run that does not declare the panel
asked for -- left out unsaid, a comparison would read as complete.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from figure.render import DASHES, Axes, Band, Line, Panel, Ref, log_refusal
from figure.style import INK_SECONDARY, NEUTRAL, series_color, shade
from record import Series, capital
from record.config import fields_of, to_dict

__all__ = ["RUNS", "COLOUR_BY", "X", "INDEX_LABEL", "chart_of", "pool_refusal", "sweep_of",
           "distinct", "telling", "series_of"]

#: How several runs share a chart: one pair of axes, a row each, or pooled by a statistic.
RUNS = ("one", "rows", "mean", "rms")
COLOUR_BY = ("run", "quantity")
#: The bottom axis: the panel's own abscissa, or the sample index, labelled ``INDEX_LABEL``.
X = ("own", "index")
INDEX_LABEL = "Sample index [-]"
_POOL_WORDS = {"mean": "mean", "rms": "RMS"}
#: How far apart runs of one colour at one value of a sweep's setting are drawn: a share of the
#: span of the setting's values (a number), or of the step between two names.
_SPREAD_NUMBER, _SPREAD_NAME = 0.012, 0.08

Pairs = Sequence[tuple[str, Any]]


def series_of(record: Any, name: str) -> Series | None:
    """The record's declared panel ``name``, or ``None`` when it declares none by that name."""
    return next((s for s in record.series() if s.name == name), None)


def _one_piece(pairs: Pairs) -> None:
    """Refuse runs of different pieces: one chart compares runs of one piece."""
    pieces = sorted({type(rec).component for _, rec in pairs})
    if len(pieces) > 1:
        raise ValueError(f"the runs are of different pieces {pieces}; one chart compares runs "
                         f"of one piece")


def _held(pairs: Pairs, name: str) -> list[tuple[str, Series]]:
    _one_piece(pairs)
    held = [(run, s) for run, rec in pairs if (s := series_of(rec, name)) is not None]
    if not held:
        raise ValueError(f"none of the runs declares a panel {name!r}")
    lacking = [run for run, rec in pairs if series_of(rec, name) is None]
    if lacking:
        raise ValueError(f"{', '.join(lacking)} declare no panel {name!r}; drawn without them, "
                         f"the comparison would leave them out unsaid")
    return held


def _choice(value: Any, allowed: Sequence[Any], what: str) -> None:
    if value not in allowed:
        raise ValueError(f"{what} is one of {list(allowed)}; {value!r} given")


def pool_refusal(pairs: Pairs, name: str, how: str) -> str:
    """Why the runs cannot be pooled by ``how`` (``"mean"`` or ``"rms"``) on panel ``name``,
    ``""`` when they can."""
    _choice(how, ("mean", "rms"), "a statistic")
    held = _held(pairs, name)
    first = held[0][1]
    if how not in first.pooled:
        return (f"the chart declares no {_POOL_WORDS[how]}" if first.pooled
                else "the chart declares no statistic")
    if len(held) < 2:
        return "one run shown"
    x0 = np.asarray(first.x, float)
    if any(np.asarray(s.x, float).shape != x0.shape or not np.array_equal(np.asarray(s.x, float), x0)
           for _, s in held):
        return "the runs are on different time grids"
    if any([label for label, _ in s.traces] != [label for label, _ in first.traces] for _, s in held):
        return "the runs draw different traces"
    return _replicate_refusal([(run, rec) for run, rec in pairs if series_of(rec, name)])


def _replicate_refusal(pairs: Pairs) -> str:
    """Why ``pairs`` are not replicates of one run whose draws can be counted together, ``""``
    when they are: they differ in nothing but their replicates (a seed) and the count of draws
    each makes from one (``draws``), and no two share a draw -- the band of all their draws
    holds for independent draws only."""
    fields = _fields(pairs)
    reps = {n: f["draws"] for n, f in fields.items() if f["replicate"]}
    if not reps:
        return "the runs declare no seed, so nothing says their draws are independent"
    free = set(reps) | {d for d in reps.values() if d}
    other = [n for n in distinct(pairs) if n not in free]
    if other:
        return (f"the runs differ in {', '.join(fields[n]['label'] for n in other)}: only runs "
                f"that differ in their {', '.join(fields[n]['label'] for n in reps)} alone pool")

    def drawn(rec: Any) -> list[set]:
        return [set(range(_value(rec, n), _value(rec, n) + _value(rec, d))) if d
                else {repr(_value(rec, n))} for n, d in reps.items()]

    for i, (a, ra) in enumerate(pairs):
        for b, rb in pairs[i + 1:]:
            both = [x & y for x, y in zip(drawn(ra), drawn(rb))]
            if all(both):
                return (f"runs {a} and {b} make the same draw: both use "
                        + ", ".join(f"{n} {min(x)}" for n, x in zip(reps, both)))
    return ""


def _x(s: Series, x: str) -> tuple[np.ndarray, str]:
    if x == "index":
        return np.arange(len(s.x), dtype=float), INDEX_LABEL
    return np.asarray(s.x, dtype=float), s.x_label


def _trace_x(s: Series, j: int, xs: np.ndarray, x: str) -> np.ndarray:
    """Trace ``j``'s abscissa: its own when the panel gives each trace one (``trace_x``, by
    sample index under ``x="index"``), else the panel's ``xs``."""
    if not s.trace_x:
        return xs
    own = np.asarray(s.trace_x[j], dtype=float)
    return np.arange(len(own), dtype=float) if x == "index" else own


def _edge(edge: Any, n: int) -> np.ndarray:
    return np.full(n, float(edge)) if np.ndim(edge) == 0 else np.asarray(edge, dtype=float)


def _band_owner(s: Series, label: str) -> int | None:
    """The trace a band is the uncertainty of: the one whose label the band's label starts with
    -- the longest such, so a band of ``x_hat`` is not ``x``'s (code review, 2026-10-27) -- else
    the only trace there is, else ``None``: a band about the panel, not one quantity (RCWS
    ``figure/draw.py``, ``_band_owner``)."""
    labels = [trace for trace, _ in s.traces]
    starts = [i for i, trace in enumerate(labels) if trace and label.startswith(trace)]
    if starts:
        return max(starts, key=lambda i: len(labels[i]))
    return 0 if len(labels) == 1 else None


def _alone(s: Series, xs: np.ndarray, points: bool, x: str = "own") -> tuple[tuple, tuple, tuple]:
    """One run's lines, bands and references: colours by position, the i-th trace slot i; a
    band its trace's colour (:func:`_band_owner`), a band about the panel the neutral."""
    n = len(xs)
    lines = tuple(Line(label, _trace_x(s, i, xs, x), np.asarray(y, float), series_color(i),
                       points=points)
                  for i, (label, y) in enumerate(s.traces))
    bands = tuple(Band(label, xs, _edge(lo, n), _edge(hi, n),
                       NEUTRAL if (owner := _band_owner(s, label)) is None else series_color(owner))
                  for label, lo, hi in s.bands)
    refs = tuple(Ref(label, float(v), INK_SECONDARY) for label, v in s.refs)
    return lines, bands, refs


def _identical(held: list[tuple[str, Series]], x: str, kind: str) -> list[str]:
    """The labels of the traces (``kind="traces"``) or bands (``"bands"``) every run in ``held``
    holds on the same abscissa with the same numbers: one source -- a truth both runs were
    scored against -- drawn once. Decided on the numbers, never on the label alone (RCWS
    ``figure/overlay.py``, ``_shared``)."""
    if len(held) < 2:
        return []

    def by_label(s: Series) -> dict[str, tuple[np.ndarray, ...]]:
        xs = _x(s, x)[0]
        if kind == "traces":
            return {label: (_trace_x(s, j, xs, x), np.asarray(y, dtype=float))
                    for j, (label, y) in enumerate(s.traces)}
        return {label: (xs, *(np.asarray(v, dtype=float) for v in rest))
                for label, *rest in [(b[0], b[1], b[2]) for b in s.bands]}

    first, *others = [by_label(s) for _, s in held]
    return [label for label, arrays in first.items()
            if all(label in o and all(a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
                                      for a, b in zip(arrays, o[label]))
                   for o in others)]


def _shared(items: list[tuple[str, tuple, Any]], runs: int) -> list[tuple[str, Any]]:
    """``(run, key, item)`` grouped by ``key``: one entry per distinct item, named by its label
    alone when every run holds it, else prefixed by the runs that do."""
    groups: dict[tuple, list] = {}
    for run, key, item in items:
        groups.setdefault(key, [item, []])[1].append(run)
    return [(("" if len(held) == runs else ", ".join(held) + " — "), item)
            for item, held in groups.values()]


def chart_of(pairs: Pairs, name: str, *, runs: str = "one", colour_by: str = "run",
             points: bool | None = None, log_y: bool | None = None, log_x: bool = False,
             x: str = "own", x_range: tuple[float, float] | None = None) -> Panel:
    """Panel ``name`` over ``pairs``, shaped by the keywords (the module); ``points`` and
    ``log_y`` start at what the panel declares (``None``) and override it when given. Refused
    in words: no run declaring the panel, runs labelling its axes differently, a statistic the
    runs cannot be pooled by (:func:`pool_refusal`), a log axis over a value at or below zero."""
    _choice(runs, RUNS, "runs"); _choice(colour_by, COLOUR_BY, "colour_by"); _choice(x, X, "x")
    held = _held(pairs, name)
    first = held[0][1]
    for what in ("x_label", "y_label"):
        said = {getattr(s, what) for _, s in held}
        if len(said) > 1:
            raise ValueError(f"the runs label panel {name!r}'s {what[0]} axis differently: "
                             f"{sorted(said)}")
    dots = bool(points) if points is not None else False
    log = first.log_y if log_y is None else bool(log_y)
    x_label = _x(first, x)[1]

    if runs in ("mean", "rms"):
        why = pool_refusal(pairs, name, runs)
        if why:
            raise ValueError(f"panel {name!r} cannot be pooled by {_POOL_WORDS[runs]}: {why}")
        xs = _x(first, x)[0]
        w = np.ones(len(held))
        lines = []
        for j, (label, _) in enumerate(first.traces):
            ys = np.array([np.asarray(s.traces[j][1], float) for _, s in held])
            y = (w @ ys) / w.sum() if runs == "mean" else np.sqrt((w @ ys ** 2) / w.sum())
            lines.append(Line(f"{capital(_POOL_WORDS[runs])} of {len(held)} runs · {label}", xs, y,
                              series_color(j), points=dots))
        bands = ()
        refs = tuple(Ref(label, float(v), INK_SECONDARY) for label, v in first.refs)
        axes = (Axes(first.y_label, tuple(lines), bands, refs, log),)
        panel = Panel(name, x_label, axes, first.doc)
    elif len(held) == 1:
        xs = _x(first, x)[0]
        lines, bands, refs = _alone(first, xs, dots, x)
        panel = Panel(name, x_label, (Axes(first.y_label, lines, bands, refs, log),), first.doc)
    elif runs == "rows":
        axes = tuple(Axes(run, *_alone(s, _x(s, x)[0], dots, x), log) for run, s in held)
        panel = Panel(name, x_label, axes, first.doc, quantity=first.y_label)
    else:
        # a run wears the slot of its place among ALL the runs given, so a view that colours
        # its runs by the order shown keeps each run's colour when one of them lacks the panel
        slot = {run: i for i, (run, _) in enumerate(pairs)}

        def colour(i: int, j: int) -> str:
            """Run ``i``'s quantity ``j``: the run's hue in the quantity's shade, or the
            transpose when colour carries the quantity (RCWS ``overlay._axes``)."""
            return shade(series_color(i), j) if colour_by == "run" else shade(series_color(j), i)

        def dash(i: int, j: int) -> str:
            return DASHES[(j if colour_by == "run" else i) % len(DASHES)]

        same_lines, same_bands = _identical(held, x, "traces"), _identical(held, x, "bands")
        # two runs are named, three or more counted: in RCWS a line 8 runs shared was named
        # with 333 characters of run names
        every = ", ".join(run for run, _ in held) if len(held) < 3 else f"{len(held)} runs"
        # the first colour slot no run -- or, with colour by quantity, no quantity -- wears
        # (code review, 2026-10-27: after the runs alone, a line every run held took the third
        # quantity's colour)
        after = (len(pairs) if colour_by == "run"
                 else max(len(s.traces) for _, s in held))
        lines, bands, ref_items = [], [], []
        for k, (run, s) in enumerate(held):
            i = slot[run]
            xs = _x(s, x)[0]
            n = len(xs)
            own = [label for label, _ in s.traces if label not in same_lines]
            for j, (label, y) in enumerate(s.traces):
                tx = _trace_x(s, j, xs, x)
                if label in same_lines:
                    if k == 0:
                        lines.append(Line(f"{every} — {label} (identical)", tx,
                                          np.asarray(y, float),
                                          series_color(after + same_lines.index(label)),
                                          points=dots))
                    continue
                lines.append(Line(run if len(own) == 1 else f"{run} — {label}", tx,
                                  np.asarray(y, float), colour(i, j), dash(i, j), dots))
            for label, lo, hi in s.bands:
                lo_, hi_ = _edge(lo, n), _edge(hi, n)
                if label in same_bands:
                    if k == 0:
                        bands.append(Band(f"{every} — {label} (identical)", xs, lo_, hi_,
                                          series_color(after + len(same_lines)
                                                       + same_bands.index(label))))
                    continue
                owner = _band_owner(s, label)
                bands.append(Band(f"{run} — {label}", xs, lo_, hi_,
                                  NEUTRAL if owner is None else colour(i, owner)))
            for r, (label, v) in enumerate(s.refs):
                ref_items.append((run, (label, float(v)),
                                  Ref(label, float(v), INK_SECONDARY, DASHES[r % len(DASHES)])))
        refs = tuple(Ref(f"{pre}{r.label}", r.value, r.color, r.dash)
                     for pre, r in _shared(ref_items, len(held)))
        panel = Panel(name, x_label,
                      (Axes(first.y_label, tuple(lines), tuple(bands), refs, log),), first.doc)
    if first.equal_axes and x == "own":       # metres against metres, not against an index
        panel = dataclasses.replace(panel, equal_axes=True)
    return _finished(panel, log_x, log, x_range)


def _finished(panel: Panel, log_x: bool, log_y: bool, x_range) -> Panel:
    if log_x:
        why = log_refusal(panel, "x")
        if why:
            raise ValueError(f"panel {panel.name!r} cannot take a logarithmic bottom axis: {why}")
    if log_y:
        why = log_refusal(panel, "y")
        if why:
            raise ValueError(f"panel {panel.name!r} cannot take a logarithmic ordinate: {why}")
    rng = None if x_range is None else (float(x_range[0]), float(x_range[1]))
    if rng is not None and not rng[0] < rng[1]:
        raise ValueError(f"an x range runs from low to high; {x_range!r} given")
    return dataclasses.replace(panel, log_x=bool(log_x), x_range=rng)


# --- a number against a setting --------------------------------------------------------------

def _fields(pairs: Pairs) -> dict[str, dict[str, Any]]:
    return {f["name"]: f for f in fields_of(type(pairs[0][1].config))}


def _value(record: Any, name: str) -> Any:
    """A setting as its record saved it: a list of rows as its dicts, which compare by what they
    hold (a row object carries an instance name of its own, so two equal lists never matched)."""
    return to_dict(record.config)[name]


def distinct(pairs: Pairs, *, but: Sequence[str] = ()) -> list[str]:
    """The settings whose saved values differ across the runs, in declared order, but ``but``."""
    out = []
    for name in _fields(pairs):
        if name not in but and len({repr(_value(rec, name)) for _, rec in pairs}) > 1:
            out.append(name)
    return out


def telling(pairs: Pairs, vs: str) -> list[str]:
    """The settings that tell the runs apart on a sweep over ``vs``: the ones that differ, but
    ``vs`` and the replicates (a seed)."""
    fields = _fields(pairs)
    return [n for n in distinct(pairs, but=(vs,)) if not fields[n]["replicate"]]


def _words(f: dict[str, Any], value: Any) -> str:
    text = f"{value:.4g}" if isinstance(value, float) else str(value)
    return f"{f['label']} = {text}" + (f" {f['unit']}" if f["unit"] else "")


def sweep_of(pairs: Pairs, number: str, *, vs: str, by: Sequence[str] | None = None,
             runs: str = "each", points: bool = True, log_x: bool = False,
             log_y: bool = False) -> Panel:
    """Summary number ``number`` against setting ``vs`` over ``pairs`` (the module). ``by`` are
    the settings each colour stands for, ``None`` for every one that tells the runs apart
    (:func:`telling`), ``()`` for one colour over every run; ``runs`` is ``"each"`` (a point per
    run) or the statistic of a colour's runs at one value (``"mean"``, ``"rms"``)."""
    _choice(runs, ("each", "mean", "rms"), "runs")
    if not pairs:
        raise ValueError("a sweep needs at least one run")
    _one_piece(pairs)
    fields = _fields(pairs)
    if vs not in fields:
        raise ValueError(f"{vs!r} is not a setting of these runs; they hold {sorted(fields)}")
    by = tuple(telling(pairs, vs) if by is None else by)
    unknown = sorted(set(by) - set(fields))
    if unknown:
        raise ValueError(f"{unknown} are not settings of these runs")
    spec = next((n for n in pairs[0][1].number_specs() if n.name == number), None)
    if spec is None:
        raise ValueError(f"the runs declare no number {number!r}")
    lacking = [run for run, rec in pairs if number not in rec.numbers]
    if lacking:
        raise ValueError(f"{lacking} carry no {number!r}")

    values = [_value(rec, vs) for _, rec in pairs]
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    # keyed by repr: a list setting (Q, R) is placed by name, as a choice is, and a list cannot
    # key a dict (TypeError, code review 2026-11-11)
    names = list(dict.fromkeys(repr(v) for v in values))
    where = ({repr(v): float(v) for v in values} if numeric
             else {repr(v): float(names.index(repr(v))) for v in values})
    xs_all = np.array([where[repr(v)] for v in values])
    # on a log bottom axis the runs at one value are spread by a share of the decades the
    # values span, so a point never lands at or below zero
    geometric = numeric and log_x and bool(np.all(xs_all > 0))
    if geometric:
        span = float(np.log10(xs_all.max() / xs_all.min())) or 1.0
    else:
        span = float(xs_all.max() - xs_all.min()) or (abs(float(xs_all.max())) or 1.0)
    step = _SPREAD_NUMBER * span if numeric else _SPREAD_NAME
    replicates = [n for n, f in fields.items() if f["replicate"]]

    colours: dict[tuple, list[tuple[str, Any]]] = {}
    for run, rec in pairs:
        colours.setdefault(tuple(repr(_value(rec, b)) for b in by), []).append((run, rec))
    lines = []
    for g, members in enumerate(colours.values()):
        label = (", ".join(_words(fields[b], _value(members[0][1], b)) for b in by)
                 or spec.label)
        at: dict[float, list[tuple[str, Any]]] = {}
        for run, rec in sorted(members, key=lambda m: [repr(_value(m[1], r)) for r in replicates]):
            at.setdefault(where[repr(_value(rec, vs))], []).append((run, rec))
        px, py, texts, spread = [], [], [], []
        for xv in sorted(at):
            here = at[xv]
            if runs != "each" and len(here) > 1:
                ys = np.array([rec.numbers[number] for _, rec in here], dtype=float)
                px.append(xv); spread.append(0.0)
                py.append(float(ys.mean() if runs == "mean" else np.sqrt(np.mean(ys ** 2))))
                texts.append(f"{capital(_POOL_WORDS[runs])} of " + ", ".join(r for r, _ in here))
                continue
            for i, (run, rec) in enumerate(here):
                px.append(xv); py.append(float(rec.numbers[number])); texts.append(run)
                k = (i - (len(here) - 1) / 2) * step
                spread.append(xv * (10.0 ** k - 1.0) if geometric else k)
        lines.append(Line(label, px, py, series_color(g), points=points, texts=tuple(texts),
                          spread=tuple(spread)))
    f = fields[vs]
    x_label = f["label"] + (f" [{f['unit']}]" if f["unit"] else "")
    y_label = spec.label + (f" [{spec.unit}]" if spec.unit else " [-]")
    shown = {repr(v): str(v) for v in values}
    ticks = None if numeric else tuple((float(i), shown[n]) for i, n in enumerate(names))
    panel = Panel(f"{number}-vs-{vs}", x_label, (Axes(y_label, tuple(lines), log_y=log_y),),
                  spec.doc, x_ticks=ticks)
    return _finished(panel, log_x, log_y, None)
