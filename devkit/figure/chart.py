"""What is on a panel: the records' own declarations, shaped once into a :class:`Panel`. BESS's
``devkit/figure/chart.py``, trimmed to two shapes: a declared panel over one or more runs on one
plot (:func:`chart_of`), and a summary number against a setting, every run a point
(:func:`sweep_of`). It never names a piece; what it cannot draw it refuses in words.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from figure.render import DASHES, Band, Line, Panel, Ref
from figure.style import INK_SECONDARY, series_color
from record import Series
from record.config import fields_of

__all__ = ["series_of", "chart_of", "sweep_of", "telling"]

Pairs = Sequence[tuple[str, Any]]     # (run name, record)


def _one_piece(pairs: Pairs) -> None:
    """Refuse runs of different pieces: one chart compares runs of one piece."""
    pieces = sorted({type(rec).component for _, rec in pairs})
    if len(pieces) > 1:
        raise ValueError(f"the runs are of different pieces {pieces}; one chart compares runs "
                         f"of one piece")


def series_of(record: Any, name: str) -> Series | None:
    """The panel ``name`` the record declares, ``None`` when it declares none by that name."""
    return next((s for s in record.series() if s.name == name), None)


def chart_of(pairs: Pairs, name: str) -> Panel:
    """Panel ``name`` over ``pairs``. One run: its traces in the series colours. Several: the
    colour carries the run (by its place in ``pairs``), the dash its trace, and the legend names
    each "run — trace" (the run alone when the panel holds one trace)."""
    _one_piece(pairs)
    held = [(i, run, s) for i, (run, rec) in enumerate(pairs) if (s := series_of(rec, name))]
    if not held:
        raise ValueError(f"no run declares the panel {name!r}")
    lacking = [run for run, rec in pairs if series_of(rec, name) is None]
    if lacking:
        raise ValueError(f"{lacking} declare no panel {name!r}; drawn without them, the "
                         f"comparison would leave them out unsaid")
    first = held[0][2]
    others = sorted({(s.x_label, s.y_label) for _, _, s in held} - {(first.x_label, first.y_label)})
    if others:
        raise ValueError(f"the runs draw {name!r} on different axes ({others} beside "
                         f"{(first.x_label, first.y_label)}); one panel holds one pair")
    lines, bands = [], []
    for i, run, s in held:
        for k, (label, y) in enumerate(s.traces):
            if len(pairs) == 1:
                lines.append(Line(label, s.x, y, series_color(k)))
            else:
                words = run if len(s.traces) == 1 else f"{run} — {label}"
                lines.append(Line(words, s.x, y, series_color(i), DASHES[k % len(DASHES)]))
        for k, (label, lo, hi) in enumerate(s.bands):
            words = label if len(pairs) == 1 else f"{run} — {label}"
            bands.append(Band(words, s.x, lo, hi, series_color(k if len(pairs) == 1 else i)))
    refs = tuple(dict.fromkeys(Ref(label, float(v), INK_SECONDARY)
                               for _, _, s in held for label, v in s.refs))
    return Panel(name, first.x_label, first.y_label, tuple(lines), tuple(bands), refs,
                 log_y=first.log_y, doc=first.doc)


def _value(record: Any, setting: str) -> Any:
    return getattr(record.config, setting)


def telling(pairs: Pairs, vs: str) -> list[str]:
    """The settings that tell the runs apart besides ``vs``, a replicate (a seed) never one: what
    a sweep's colours stand for."""
    fields = fields_of(type(pairs[0][1].config))
    return [f["name"] for f in fields
            if f["name"] != vs and not f["replicate"]
            and len({repr(_value(rec, f["name"])) for _, rec in pairs}) > 1]


def _unit(unit: str) -> str:
    return f" [{unit or '-'}]"


def sweep_of(pairs: Pairs, number: str, vs: str, by: Sequence[str] | None = None) -> Panel:
    """Summary number ``number`` against setting ``vs``, every run a point. ``by`` names the
    settings each colour stands for: ``None`` for every one that tells the runs apart besides
    ``vs`` (:func:`telling`), ``()`` for one colour over every run. A setting holding names, or
    left at the code's own default (``None``), is placed by name along the bottom axis."""
    if not pairs:
        raise ValueError("a sweep needs at least one run")
    _one_piece(pairs)
    fields = {f["name"]: f for f in fields_of(type(pairs[0][1].config))}
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
    names = list(dict.fromkeys(repr(v) for v in values))
    where = (lambda v: float(v)) if numeric else (lambda v: float(names.index(repr(v))))

    groups: dict[tuple, list[tuple[str, Any]]] = {}
    for run, rec in pairs:
        groups.setdefault(tuple(repr(_value(rec, b)) for b in by), []).append((run, rec))
    lines = []
    for g, members in enumerate(groups.values()):
        label = ", ".join(f"{b}={_value(members[0][1], b)}" for b in by) or spec.label
        ordered = sorted(members, key=lambda m: where(_value(m[1], vs)))
        lines.append(Line(label, [where(_value(rec, vs)) for _, rec in ordered],
                          [float(rec.numbers[number]) for _, rec in ordered],
                          series_color(g), points=True))
    ticks = None if numeric else tuple((float(i), w) for i, w in enumerate(names))
    return Panel(f"{number}-vs-{vs}", fields[vs]["label"] + _unit(fields[vs]["unit"]),
                 spec.label + _unit(spec.unit), tuple(lines), doc=spec.doc, x_ticks=ticks)
