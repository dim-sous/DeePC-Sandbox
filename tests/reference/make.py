"""Write the safety net's frozen files: ``<case>.npz`` per case, ``metrics-<plant>.json`` per
metrics case and ``manifest.json`` (each case's conditions, the commit and the library versions).

    uv run python -m tests.reference.make

Run it only on a branch that means to move the numbers, and say so in that branch's PR:
``tests/test_reference.py`` compares today's code against these files bit for bit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np

from tests.reference import cases

#: The libraries whose builds the numbers depend on, recorded so a mismatch on another
#: computer can be read against them.
LIBRARIES = ("numpy", "scipy", "cvxpy", "osqp", "clarabel", "scs")


def _git(*args: str) -> str:
    # from the repository root: the paths in cases.CODE are the root's (a run from this folder
    # matched none of them, and the manifest named no changed code whatever had changed)
    return subprocess.run(["git", *args], cwd=cases.ROOT, check=True,
                          capture_output=True, text=True).stdout.rstrip("\n")


def _code_sha256() -> str:
    """One fingerprint over the contents of every Python file in ``cases.CODE``, in path order:
    it names the code the numbers came from even when the commit does not."""
    digest = hashlib.sha256()
    for entry in cases.CODE:
        path = cases.ROOT / entry
        for f in sorted(path.rglob("*.py")) if path.is_dir() else [path]:
            digest.update(str(f.relative_to(cases.ROOT)).encode() + b"\0" + f.read_bytes())
    return digest.hexdigest()


def main() -> None:
    # A case taken out of the registry takes its frozen file with it; the test refuses any
    # file or manifest entry with no case.
    npz, metrics_files = cases.frozen_names()
    for stem in npz - set(cases.CASES):
        (cases.REFERENCE_DIR / f"{stem}.npz").unlink()
        print(f"removed {stem}.npz")
    for stem in metrics_files - set(cases.METRICS):
        (cases.REFERENCE_DIR / f"{stem}.json").unlink()
        print(f"removed {stem}.json")
    for name in cases.CASES:
        np.savez(cases.REFERENCE_DIR / f"{name}.npz", **cases.produce(name))
        print(f"wrote {name}.npz")
    for name in cases.METRICS:
        metrics = cases.metrics_of(name)
        (cases.REFERENCE_DIR / f"{name}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"wrote {name}.json")
    manifest = {
        "made": {"git_sha": _git("rev-parse", "--short", "HEAD"),
                 "git_dirty": bool(_git("status", "--porcelain")),
                 # the code's own files that differ from that commit, and their fingerprint
                 "code_changed": _git("status", "--porcelain", "--", *cases.CODE).splitlines(),
                 "code_sha256": _code_sha256(),
                 "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "versions": {lib: version(lib) for lib in LIBRARIES}},
        "cases": {name: cases.conditions_of(name) for name in cases.CASES},
        "metrics": {name: {**cases.metrics_conditions(name),
                           "dropped": "every *solve_time_s metric (wall clock)"}
                    for name in cases.METRICS},
    }
    cases.MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote manifest.json")


if __name__ == "__main__":
    main()
