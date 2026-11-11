"""One panel spec, drawn as a PNG still with matplotlib (the Lab's live twin will draw the same
spec). BESS's ``devkit/figure/render.py``, trimmed: one pair of axes per panel.

:mod:`figure.chart` decides WHAT is on a panel -- which run wears which colour, which trace which
dash -- and says it once as a :class:`Panel`; the backend draws that spec and adds nothing of its
own. It never titles a figure, never chooses a colour, and draws a legend only when there is more
than one label (``tests/test_figure.py``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from figure.style import INK_SECONDARY, LINE_WIDTH, REF_WIDTH, apply_style, save_figure

__all__ = ["Line", "Band", "Ref", "Panel", "DASHES", "BAND_ALPHA", "SHORT", "labels",
           "has_legend", "render"]

#: Dash patterns, in order, for what the colour does not tell apart (a run's traces when the
#: colour carries the run).
DASHES: tuple[str, ...] = ("-", "--", ":", "-.", (0, (8, 3)))
#: A band's fill (RCWS's ``render.BAND_ALPHA``).
BAND_ALPHA = 0.18
#: Below this many samples a line also marks its samples (RCWS ``render.SHORT``).
SHORT = 24


@dataclasses.dataclass(frozen=True)
class Line:
    """A trace on its own abscissa: two runs on one axes may hold different grids. ``points``
    draws markers only (a sweep's runs)."""
    label: str
    x: Any
    y: Any
    color: str
    dash: Any = "-"
    points: bool = False


@dataclasses.dataclass(frozen=True)
class Band:
    label: str
    x: Any
    lo: Any
    hi: Any
    color: str


@dataclasses.dataclass(frozen=True)
class Ref:
    """A reference line: a quantity, named like any trace."""
    label: str
    value: float
    color: str


@dataclasses.dataclass(frozen=True)
class Panel:
    """A figure: its file stem, its axis words (each with its unit), what is on it, in drawing
    order (bands, references, lines). ``x_ticks`` names the positions of a bottom axis of names."""
    name: str
    x_label: str
    y_label: str
    lines: tuple[Line, ...] = ()
    bands: tuple[Band, ...] = ()
    refs: tuple[Ref, ...] = ()
    log_y: bool = False
    doc: str = ""
    x_ticks: tuple[tuple[float, str], ...] | None = None


def labels(panel: Panel) -> list[str]:
    """Every label the panel's legend would name, in drawing order, each once."""
    return list(dict.fromkeys([b.label for b in panel.bands] + [r.label for r in panel.refs]
                              + [ln.label for ln in panel.lines]))


def has_legend(panel: Panel) -> bool:
    """A legend only when there is more than one thing to tell apart; one trace is named by its
    axis labels and the file name."""
    return len(labels(panel)) > 1


def render(panel: Panel, out_dir: Path, prov, *, facts: dict[str, Any] | None = None) -> list[Path]:
    """Draw ``panel`` as ``<out_dir>/<panel.name>.png`` with its sidecar."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    apply_style(fig, ax)
    for b in panel.bands:
        ax.fill_between(np.asarray(b.x, dtype=float), b.lo, b.hi, color=b.color,
                        alpha=BAND_ALPHA, linewidth=0, label=b.label)
    for r in panel.refs:
        ax.axhline(r.value, color=r.color, linewidth=REF_WIDTH, linestyle="--", label=r.label)
    for ln in panel.lines:
        x, y = np.asarray(ln.x, dtype=float), np.asarray(ln.y, dtype=float)
        if ln.points:
            ax.plot(x, y, linestyle="none", marker="o", markersize=5, color=ln.color, label=ln.label)
        else:
            ax.plot(x, y, color=ln.color, linestyle=ln.dash, linewidth=LINE_WIDTH,
                    marker="o" if len(x) < SHORT else None, markersize=3, label=ln.label)
    ax.set_xlabel(panel.x_label)
    ax.set_ylabel(panel.y_label)
    if panel.log_y:
        ax.set_yscale("log")
    if panel.x_ticks:
        ax.set_xticks([p for p, _ in panel.x_ticks], [w for _, w in panel.x_ticks])
    if has_legend(panel):
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return [save_figure(fig, out_dir, panel.name, facts or {}, prov)]
