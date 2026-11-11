"""Tiny real records for the Lab's tests, written by the terminal itself (BESS's
``tests/lab_runs.py``, adapted): the coupled masses' default dataset run, and two 1 s closed loops
on it, scenario ``sinusoidal`` (the ``default`` steps hold zero for their first 5 s), that differ
in λ_g alone -- about four seconds each, short enough to be cheap and long enough to hold twenty
solves. λ_g = 2 for the second: at 50, OSQP's solves averaged 0.78 s against 0.20 s at the
default 10, and the run took 15.5 s (measured 2026-11-11, this file's runs)."""

from __future__ import annotations

import os
from pathlib import Path

import terminal

OUT_DIR = "results"


def make_runs(root: Path) -> tuple[str, str]:
    """Write the dataset run ``default`` under ``<root>/results/dataset/``, where a run the Lab
    launches from ``root`` finds it by the closed loop's default, then ``first`` (every setting at
    its default but the scenario) and ``second`` (also ``lambda_g=2``) on it under
    ``<root>/results/closed_loop/``, the first one older by the clock; return ``(root, out_dir)``
    as the Lab takes them."""
    out = root / OUT_DIR
    terminal.main(["run", "dataset", "--name", "default", "--out-dir", str(out)])
    data = out / "dataset" / "default.json"
    base = ["run", "closed_loop", "--set", f"dataset_run={data}", "--set", "sim_duration=1.0",
            "--set", "scenario=sinusoidal", "--out-dir", str(out)]
    terminal.main(base + ["--name", "first"])
    terminal.main(base + ["--set", "lambda_g=2", "--name", "second"])
    first, second = out / "closed_loop" / "first.json", out / "closed_loop" / "second.json"
    now = second.stat().st_mtime
    os.utime(first, (now - 60, now - 60))            # the clock can step; pin the order
    return str(root), OUT_DIR
