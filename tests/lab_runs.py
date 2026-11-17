"""Tiny real records for the Lab's tests, written by the command line itself (BESS's
``tests/lab_runs.py``, adapted): the default system run, the default open-loop run on it, and two
1 s closed loops on that, that differ in λ_g alone -- 20 solves each after the T_ini = 4 warm-up,
short enough to be cheap. λ_g = 2 for the second (a value carried from the old file's loops,
where λ_g = 50 made OSQP's solves about four times slower, measured 2026-11-11)."""

from __future__ import annotations

import os
from pathlib import Path

from benchlab.cli import main

OUT_DIR = "results"


def make_runs(root: Path) -> tuple[str, str]:
    """Write the system run ``default`` and the open-loop run ``default`` on it under
    ``<root>/results/``, where a run the Lab launches from ``root`` finds them by the next piece's
    default, then ``first`` (every setting at its default but the duration) and ``second`` (also
    ``lambda_g=2``) under ``<root>/results/closed_loop/``, the first one older by the clock; return
    ``(root, out_dir)`` as the Lab takes them."""
    out = root / OUT_DIR
    main(["run", "system", "--name", "default", "--out-dir", str(out)])
    system = out / "system" / "default.json"
    main(["run", "open_loop", "--set", f"system_run={system}", "--name", "default",
          "--out-dir", str(out)])
    data = out / "open_loop" / "default.json"
    base = ["run", "closed_loop", "--set", f"open_loop_run={data}", "--set", "duration=1.0",
            "--out-dir", str(out)]
    main(base + ["--name", "first"])
    main(base + ["--set", "lambda_g=2", "--name", "second"])
    first, second = out / "closed_loop" / "first.json", out / "closed_loop" / "second.json"
    now = second.stat().st_mtime
    os.utime(first, (now - 60, now - 60))            # the clock can step; pin the order
    return str(root), OUT_DIR
