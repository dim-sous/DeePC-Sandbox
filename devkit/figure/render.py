"""One panel spec, three artefacts: a PNG still (matplotlib, here), the Lab's live twin (plotly,
:mod:`figure.twin`) and, asked, the numbers on it as a table (:func:`table_of`, a CSV beside
the still).

:mod:`figure.chart` decides WHAT is on a panel -- which run wears which colour, which trace
which dash, what is pooled over the runs -- and says it once as a :class:`Panel`; the backends
draw that spec and add nothing of their own, so a still, the page's chart and the table cannot
disagree. Neither backend titles a figure or chooses a colour (``tests/test_figure.py``).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import numpy as np

from figure.style import (INK_SECONDARY, LINE_WIDTH, REF_WIDTH, apply_style, save_figure,
                          save_table)

__all__ = ["Line", "Band", "Ref", "Axes", "Panel", "Table", "DASHES", "BAND_ALPHA", "SHORT",
           "broken", "has_legend", "log_refusal", "render", "table_of"]

#: Dash patterns, in order, for what the colour does not tell apart (the traces of one run when
#: colour carries the run, the runs when it carries the quantity). Plotly's names; the still
#: translates them.
DASHES: tuple[str, ...] = ("solid", "dash", "dot", "dashdot", "longdash")
_MPL_DASH = {"solid": "-", "dash": "--", "dot": ":", "dashdot": "-.", "longdash": (0, (8, 3))}
#: A band's fill: RCWS's ``render.BAND_ALPHA``, chosen there so a line under a band stays
#: readable and two runs' bands are still told apart.
BAND_ALPHA = 0.18
#: Below this many samples a line also marks its samples (RCWS ``render.SHORT``).
SHORT = 24
#: A legend name longer than this is broken onto lines at its own separators (:func:`broken`).
#: RCWS ``render.LEGEND_LINE``: a number card over 8 runs named its curves with 715-905
#: characters each, and a horizontal legend of them ran off the card's left edge.
LEGEND_LINE = 60
#: Where a name may break: after the dash joining a run to its trace, the dot joining one
#: setting to the next, the comma listing runs.
_BREAK_AFTER = re.compile(r"(?<= — )|(?<= · )|(?<=, )")


def broken(label: str, newline: str = "\n") -> str:
    """``label`` on lines of at most :data:`LEGEND_LINE` characters, broken after its own
    separators and joined by ``newline``: every word kept, a piece longer than a line kept
    whole, and a label that fits returned as it is."""
    if len(label) <= LEGEND_LINE:
        return label
    lines, line = [], ""
    for piece in _BREAK_AFTER.split(label):
        if line and len(line) + len(piece) > LEGEND_LINE:
            lines.append(line.rstrip())
            line = piece
        else:
            line += piece
    lines.append(line.rstrip())
    return newline.join(lines)


@dataclasses.dataclass(frozen=True)
class Line:
    """A trace on its own abscissa: two runs on one axes may hold different time grids.
    ``texts`` names each point (the run a sweep drew it from) for a hover; ``spread`` moves each
    point to the side by that much where several share one abscissa value -- the table keeps the
    abscissa itself."""
    label: str
    x: Any
    y: Any
    color: str
    dash: str = "solid"
    points: bool = False
    texts: tuple[str, ...] = ()
    spread: tuple[float, ...] = ()

    def drawn_x(self) -> np.ndarray:
        x = np.asarray(self.x, dtype=float)
        return x + np.asarray(self.spread, dtype=float) if self.spread else x


@dataclasses.dataclass(frozen=True)
class Band:
    label: str
    x: Any
    lo: Any
    hi: Any
    color: str


@dataclasses.dataclass(frozen=True)
class Ref:
    label: str
    value: float
    color: str
    dash: str = "solid"


@dataclasses.dataclass(frozen=True)
class Axes:
    """One pair of axes, one ordinate unit: what is on it, in drawing order (bands, references,
    lines). ``y_label`` is the ordinate's words -- the quantity with its unit, or, on a row of a
    panel stacked one row per run, the run's name, the quantity then being the panel's."""
    y_label: str
    lines: tuple[Line, ...] = ()
    bands: tuple[Band, ...] = ()
    refs: tuple[Ref, ...] = ()
    log_y: bool = False


@dataclasses.dataclass(frozen=True)
class Panel:
    """A figure: its file stem, its bottom axis and its rows (one, or one per run). ``quantity``
    is the ordinate's words shared by every row of a stacked panel, ``""`` for one row.
    ``x_ticks`` names the positions of a bottom axis of names; ``x_range`` is the span shown,
    ``None`` for all of it."""
    name: str
    x_label: str
    axes: tuple[Axes, ...]
    doc: str = ""
    quantity: str = ""
    log_x: bool = False
    x_range: tuple[float, float] | None = None
    x_ticks: tuple[tuple[float, str], ...] | None = None


def _labels(panel: Panel) -> set[str]:
    return {item.label for ax in panel.axes for item in (*ax.lines, *ax.bands, *ax.refs)}


def has_legend(panel: Panel) -> bool:
    """A legend only when there is more than one named thing to tell apart; one trace is named
    by the axis labels and the file name."""
    return len(_labels(panel)) > 1


def log_refusal(panel: Panel, axis: str) -> str:
    """Why ``panel`` cannot be drawn on a logarithmic ``axis`` (``"x"`` or ``"y"``), ``""`` when
    it can: a log axis drops a value at or below zero without a word, so it is refused here."""
    if axis not in ("x", "y"):
        raise ValueError(f"axis is 'x' or 'y'; {axis!r} given")
    if axis == "x" and panel.x_ticks:
        return "its bottom axis holds names"
    if axis == "x":
        # the values, not where a sweep spreads its points: on a log axis it spreads them by a
        # share of a decade (`chart.sweep_of`), which keeps a positive value positive
        held = [np.asarray(line.x, dtype=float) for ax in panel.axes for line in ax.lines]
    else:
        held = [np.asarray(v, dtype=float) for ax in panel.axes
                for v in [*(line.y for line in ax.lines), *(b.lo for b in ax.bands),
                          *(b.hi for b in ax.bands), *([r.value] for r in ax.refs)]]
    values = np.concatenate([np.ravel(v) for v in held]) if held else np.zeros(0)
    low = values[np.isfinite(values) & (values <= 0.0)]
    if not len(low):
        return ""
    what = "its bottom axis holds" if axis == "x" else "it holds"
    return f"{what} {len(np.unique(low)) if axis == 'x' else len(low)} value(s) at or below zero"


# --- the still -------------------------------------------------------------------------------

def render(panel: Panel, out_dir: Path, prov, *, facts: dict[str, Any] | None = None,
           table: bool = False) -> list[Path]:
    """``<out_dir>/<panel>.png`` with its sidecar, and with ``table`` the numbers on it as
    ``<panel>.csv`` beside it. ``facts`` (what shaped the drawing) ride in the sidecars."""
    import matplotlib.pyplot as plt

    n = len(panel.axes)
    fig, axs = plt.subplots(n, 1, sharex=True, squeeze=False,
                            figsize=(9.0, 4.2 if n == 1 else 2.3 * n + 0.9))
    for ax, spec in zip(axs[:, 0], panel.axes):
        apply_style(fig, ax)
        for band in spec.bands:
            ax.fill_between(np.asarray(band.x, float), np.asarray(band.lo, float) * np.ones(len(band.x)),
                            np.asarray(band.hi, float) * np.ones(len(band.x)), color=band.color,
                            alpha=BAND_ALPHA, linewidth=0, label=broken(band.label))
        span = _span(spec)
        for ref in spec.refs:
            ax.plot(span, [ref.value, ref.value], color=ref.color, linewidth=REF_WIDTH,
                    linestyle=_MPL_DASH[ref.dash], label=broken(ref.label))
        for line in spec.lines:
            style = ({"marker": "o", "ms": 3.5, "linestyle": "none"} if line.points else
                     {"linestyle": _MPL_DASH[line.dash], "solid_joinstyle": "round",
                      "solid_capstyle": "round",
                      **({"marker": "o", "ms": 3.5} if len(line.x) < SHORT else {})})
            ax.plot(line.drawn_x(), np.asarray(line.y, float), color=line.color,
                    linewidth=LINE_WIDTH, label=broken(line.label), **style)
        if spec.log_y:
            ax.set_yscale("log", nonpositive="clip")
        ax.set_ylabel(spec.y_label)
    bottom = axs[-1, 0]
    if panel.log_x:
        bottom.set_xscale("log")
    if panel.x_ticks:
        bottom.set_xticks([p for p, _ in panel.x_ticks], [w for _, w in panel.x_ticks])
    if panel.x_range is not None:
        bottom.set_xlim(*panel.x_range)
    bottom.set_xlabel(panel.x_label)
    if panel.quantity:
        fig.supylabel(panel.quantity, color=INK_SECONDARY, fontsize=10)
    if has_legend(panel):
        handles, labels, seen = [], [], set()
        for ax in axs[:, 0]:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in seen:
                    seen.add(lab); handles.append(h); labels.append(lab)
        axs[0, 0].legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    facts = {"panel": panel.name, "doc": panel.doc, "x_label": panel.x_label,
             "y_label": panel.quantity or panel.axes[0].y_label,
             "xlim": list(map(float, bottom.get_xlim())),
             "ylim": [list(map(float, ax.get_ylim())) for ax in axs[:, 0]],
             "log_y": [ax.log_y for ax in panel.axes], "log_x": panel.log_x,
             "traces": [line.label for ax in panel.axes for line in ax.lines], **(facts or {})}
    paths = [save_figure(fig, out_dir, panel.name, facts, prov)]
    if table:
        paths.append(save_table(table_of(panel), out_dir, panel.name, facts, prov))
    return paths


def _span(spec: Axes) -> list[float]:
    xs = [line.drawn_x() for line in spec.lines] + [np.asarray(b.x, float) for b in spec.bands]
    finite = np.concatenate([x[np.isfinite(x)] for x in xs]) if xs else np.array([0.0, 1.0])
    return [float(finite.min()), float(finite.max())]


# --- the numbers on it -----------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Table:
    """What a panel draws, as numbers: ``header`` and ``rows`` (every point of every trace, and
    of a band that varies), ``beside`` the reference quantities and the constant bands, each
    ``(words, values)``. A value is a float, or the words naming a point."""
    header: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    beside: tuple[tuple[str, tuple[float, ...]], ...]


def table_of(panel: Panel) -> Table:
    """The numbers ``panel`` draws, read off its spec: one column per trace on a shared abscissa
    (a row of a stacked panel prefixing its traces with the row's words), else one row per
    point. Nothing is computed here but the layout of what the spec holds."""
    columns: list[tuple[str, Line | tuple[str, Any, Any]]] = []
    beside: list[tuple[str, tuple[float, ...]]] = []
    stacked = len(panel.axes) > 1
    for ax in panel.axes:
        prefix = f"{ax.y_label} · " if stacked else ""
        for line in ax.lines:
            columns.append((prefix + line.label, line))
        for band in ax.bands:
            lo, hi = np.ravel(np.asarray(band.lo, float)), np.ravel(np.asarray(band.hi, float))
            if lo.size == 1 or (np.all(lo == lo[0]) and np.all(hi == hi[0])):
                beside.append((prefix + band.label, (float(lo[0]), float(hi[0]))))
            else:
                columns.append((prefix + band.label + " (lower)", (band.label, band.x, lo)))
                columns.append((prefix + band.label + " (upper)", (band.label, band.x, hi)))
        for ref in ax.refs:
            beside.append((prefix + ref.label, (float(ref.value),)))
    xs = [np.asarray(c.x if isinstance(c, Line) else c[1], float) for _, c in columns]
    texts = any(isinstance(c, Line) and c.texts for _, c in columns)
    if xs and not texts and all(x.shape == xs[0].shape and np.array_equal(x, xs[0]) for x in xs):
        ys = [np.asarray(c.y if isinstance(c, Line) else c[2], float) for _, c in columns]
        rows = tuple((float(xs[0][i]), *(float(y[i]) for y in ys)) for i in range(len(xs[0])))
        return Table((panel.x_label, *(w for w, _ in columns)), rows, tuple(beside))
    long: list[tuple[Any, ...]] = []
    for words, c in columns:
        x = c.x if isinstance(c, Line) else c[1]
        y = c.y if isinstance(c, Line) else c[2]
        names = c.texts if isinstance(c, Line) and c.texts else ("",) * len(np.ravel(x))
        long += [(words, name, float(a), float(b)) for name, a, b in zip(names, np.ravel(x), np.ravel(y))]
    ticks = dict(panel.x_ticks or ())
    if ticks:
        long = [(w, n, ticks.get(a, a), b) for w, n, a, b in long]
    return Table(("trace", "point", panel.x_label, panel.quantity or panel.axes[0].y_label),
                 tuple(long), tuple(beside))
