"""Records -> stills: each declared panel over one or more runs, or one number against a setting.
Each draw stamps provenance once."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from figure.chart import chart_of, series_of, sweep_of
from figure.render import render
from record import stamp

__all__ = ["draw", "draw_sweep"]


def draw(pairs: Sequence[tuple[str, Any]], out_dir: Path, *,
         panels: Sequence[str] | None = None) -> list[Path]:
    """``panels`` (every panel the first record declares, when ``None``) over ``pairs``."""
    names = list(panels) if panels is not None else [s.name for s in pairs[0][1].series()]
    unknown = [n for n in names if not any(series_of(rec, n) for _, rec in pairs)]
    if unknown:
        raise ValueError(f"no run declares the panel(s) {unknown}")
    prov, facts = stamp(0.0), {"runs": [run for run, _ in pairs]}
    out: list[Path] = []
    for name in names:
        out += render(chart_of(pairs, name), Path(out_dir), prov, facts=facts)
    return out


def draw_sweep(pairs: Sequence[tuple[str, Any]], out_dir: Path, number: str, vs: str,
               by: Sequence[str] | None = None) -> list[Path]:
    """Number ``number`` against setting ``vs`` over ``pairs``, as
    ``<out_dir>/<number>-vs-<vs>.png``."""
    facts = {"runs": [run for run, _ in pairs], "number": number, "vs": vs,
             "by": None if by is None else list(by)}
    return render(sweep_of(pairs, number, vs, by), Path(out_dir), stamp(0.0), facts=facts)
