"""Records -> stills: one record's every panel, or the panels of several shaped as a view
showed them, or one number against a setting. Each draw stamps provenance once."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from figure.chart import chart_of, series_of, sweep_of
from figure.render import render
from record import Record, stamp

__all__ = ["draw_one", "draw", "draw_sweep"]


def draw_one(record: Record, out_dir: Path) -> list[Path]:
    """Every ``Series`` the record declares, as ``<out_dir>/<panel>.png`` with its sidecar."""
    return draw([(record.name, record)], out_dir)


def draw(pairs: Sequence[tuple[str, Any]], out_dir: Path, *, panels: Sequence[str] | None = None,
         **shaping: Any) -> list[Path]:
    """``panels`` (every panel the first record declares, when ``None``) over ``pairs``, each
    shaped by ``shaping`` (:func:`figure.chart.chart_of`'s keywords: ``runs``)."""
    names = list(panels) if panels is not None else [s.name for s in pairs[0][1].series()]
    unknown = [n for n in names if not any(series_of(rec, n) for _, rec in pairs)]
    if unknown:
        raise ValueError(f"no run declares the panel(s) {unknown}")
    prov = stamp(0.0)
    facts = {"runs": [run for run, _ in pairs],
             "shaping": {k: v for k, v in shaping.items() if v is not None}}
    out: list[Path] = []
    for name in names:
        out += render(chart_of(pairs, name, **shaping), Path(out_dir), prov, facts=facts)
    return out


def draw_sweep(pairs: Sequence[tuple[str, Any]], out_dir: Path, number: str,
               **shaping: Any) -> list[Path]:
    """Number ``number`` against a setting over ``pairs`` (:func:`figure.chart.sweep_of`'s
    keywords), as ``<out_dir>/<number>-vs-<setting>.png``."""
    facts = {"runs": [run for run, _ in pairs], "number": number,
             "shaping": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in shaping.items() if v is not None}}
    return render(sweep_of(pairs, number, **shaping), Path(out_dir), stamp(0.0), facts=facts)
