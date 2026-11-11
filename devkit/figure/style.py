"""Colours, surfaces and the one save chokepoint. BESS's ``devkit/figure/style.py`` (itself RCWS's
look), trimmed: colour goes by position in the sequence the caller gives -- the runs, or a run's
traces -- and cycles past the eighth. Text wears ink, never a series colour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SERIES: tuple[str, ...] = ("#3b6db5", "#c0392b", "#1f7a3a", "#c2571a", "#7b4ea8", "#333333",
                           "#e08a1e", "#6a3d9a")
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LINE_WIDTH, REF_WIDTH, DPI = 1.6, 1.0, 150


def series_color(i: int) -> str:
    """The colour of the ``i``-th of a sequence the caller orders (runs, traces). Cycles."""
    return SERIES[int(i) % len(SERIES)]


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
