"""Tiny real records for the Lab's tests, written by the terminal itself (BESS's
``tests/lab_runs.py``, adapted): the bicycle's default dataset run, and two 1 s closed loops on it
that differ in λ_g alone -- about four seconds each, short enough to be cheap and long enough to
hold ten solves."""

from __future__ import annotations

import os
from pathlib import Path

import terminal

OUT_DIR = "results"


def make_runs(root: Path) -> tuple[str, str]:
    """Write the dataset run ``default`` under ``<root>/results/dataset/``, where a run the Lab
    launches from ``root`` finds it by the closed loop's default, then ``first`` (every setting at
    its default) and ``second`` (``lambda_g=50``) on it under ``<root>/results/closed_loop/``, the
    first one older by the clock; return ``(root, out_dir)`` as the Lab takes them."""
    out = root / OUT_DIR
    terminal.main(["run", "dataset", "--name", "default", "--out-dir", str(out)])
    data = out / "dataset" / "default.json"
    base = ["run", "closed_loop", "--set", f"dataset_run={data}", "--set", "sim_duration=1.0",
            "--out-dir", str(out)]
    terminal.main(base + ["--name", "first"])
    terminal.main(base + ["--set", "lambda_g=50", "--name", "second"])
    first, second = out / "closed_loop" / "first.json", out / "closed_loop" / "second.json"
    now = second.stat().st_mtime
    os.utime(first, (now - 60, now - 60))            # the clock can step; pin the order
    return str(root), OUT_DIR
