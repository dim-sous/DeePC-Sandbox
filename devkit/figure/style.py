"""Colours, surfaces, type and the one save chokepoint.

The categorical hues are RCWS's (``figure/style.py`` on ``feat/unequal-state-mixing``,
822b0cf), taken with the rest of its look on 2026-10-27, the developer's choice: colour goes by
position in the sequence the caller gives -- the runs, or a run's quantities -- and cycles past
the eighth; the other of the two, when a chart tells both apart, is a shade of that hue
(:func:`shade`). Text wears ink, never a series colour. The page's paper, inks and type are
these too (``lab.skin``), so a chart sits on the same paper as its card.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SERIES: tuple[str, ...] = ("#3b6db5", "#c0392b", "#1f7a3a", "#c2571a", "#7b4ea8", "#333333",
                           "#e08a1e", "#6a3d9a")
SURFACE, INK, INK_SECONDARY, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c3c2b7"
LINE_WIDTH, REF_WIDTH, DPI = 1.6, 1.0, 150
#: The type the page loads (IBM Plex); anywhere it is not loaded the browser falls back along
#: the list.
FONT = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
MONO = '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'


def series_color(i: int) -> str:
    """The colour of the ``i``-th of a sequence the caller orders (runs, quantities). Cycles."""
    return SERIES[int(i) % len(SERIES)]


def _hsl(hex_color: str) -> tuple[float, float, float]:
    """``(hue 0-360, saturation, lightness)`` of a ``#rrggbb``."""
    raw = hex_color.lstrip("#")
    r, g, b = (int(raw[i:i + 2], 16) / 255 for i in (0, 2, 4))
    high, low = max(r, g, b), min(r, g, b)
    spread = high - low
    if not spread:
        hue = 0.0
    elif high == r:
        hue = ((g - b) / spread) % 6
    elif high == g:
        hue = (b - r) / spread + 2
    else:
        hue = (r - g) / spread + 4
    light = (high + low) / 2
    sat = spread / (1 - abs(2 * light - 1)) if spread else 0.0
    return hue * 60, sat, light


def _from_hsl(hue: float, sat: float, light: float) -> str:
    """``#rrggbb`` for a hue in degrees and a saturation and lightness in 0..1."""
    chroma = (1 - abs(2 * light - 1)) * sat
    second = chroma * (1 - abs((hue / 60) % 2 - 1))
    base = [(chroma, second, 0.0), (second, chroma, 0.0), (0.0, chroma, second),
            (0.0, second, chroma), (second, 0.0, chroma), (chroma, 0.0, second)]
    r, g, b = base[int(hue // 60) % 6]
    lift = light - chroma / 2
    return "#" + "".join(f"{round(max(0.0, min(1.0, v + lift)) * 255):02x}" for v in (r, g, b))


def shade(color: str, k: int) -> str:
    """The ``k``-th shade of ``color``: the same hue, its lightness stepped; ``k=0`` is ``color``.
    When colour carries the run, the run's traces are its shades, so each run stays one hue
    while its traces are told apart -- RCWS: "four traces of one run drew as four dashes of one
    blue and could not be read" by dash alone."""
    k = int(k)
    if not k:
        return color
    hue, sat, light = _hsl(color)
    step = 0.15 * ((k + 1) // 2) * (1 if k % 2 else -1)
    return _from_hsl(hue, max(0.20, sat - 0.05 * k), min(0.80, max(0.26, light + step)))


def apply_style(fig, ax) -> None:
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def save_figure(fig, out_dir: Path, stem: str, facts: dict[str, Any], prov) -> Path:
    """The only ``savefig`` in the layer: the PNG plus a JSON sidecar saying what produced it.
    ``prov`` is one ``Provenance`` stamped per draw, shared by every panel of it."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    sidecar = {"figure": path.name, "git_sha": prov.git_sha, "git_dirty": prov.git_dirty,
               "date": prov.date, "command": " ".join(sys.argv), **facts}
    (out_dir / f"{stem}.png.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    return path
