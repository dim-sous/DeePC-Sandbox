"""DeePC plugs into Benchlab: the one check Benchlab ships for a project (its ``README.md``, "What
a project may use"), on a small run of every piece, made through the command line."""

from __future__ import annotations

from pathlib import Path

from benchlab.testing import check_project

from tests.lab_runs import make_runs


def test_every_piece_plugs_into_the_bench(tmp_path) -> None:
    root, out_dir = make_runs(tmp_path)
    out = Path(root) / out_dir
    check_project(runs=[out / "system" / "default.json", out / "open_loop" / "default.json",
                        out / "closed_loop" / "first.json", out / "closed_loop" / "second.json"])
