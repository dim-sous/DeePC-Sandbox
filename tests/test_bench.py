"""DeePC plugs into Benchlab: the one check Benchlab ships for a project (its ``README.md``, "What
a project may use"), on a small run of every piece, made through the command line."""

from __future__ import annotations

from pathlib import Path

from benchlab.cli import main
from benchlab.testing import check_project

from tests.lab_runs import make_runs


def test_every_piece_plugs_into_the_bench(tmp_path) -> None:
    root, out_dir = make_runs(tmp_path)
    out = Path(root) / out_dir
    system = out / "system" / "default.json"
    main(["run", "system", "--name", "default", "--out-dir", str(out)])
    main(["run", "open_loop", "--set", f"system_run={system}", "--name", "default",
          "--out-dir", str(out)])
    check_project(runs=[system, out / "open_loop" / "default.json",
                        out / "dataset" / "default.json", out / "closed_loop" / "first.json",
                        out / "closed_loop" / "second.json"])
