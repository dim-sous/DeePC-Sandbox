"""Write the safety net's frozen files: ``<case>.npz`` per case and ``manifest.json`` (each case's
conditions, what each piece leaves out, the commit, the code's fingerprint and the computer).

    uv run python -m tests.reference.make

Run it only on a branch that means to move the numbers, and say so in that branch's PR; or on a
computer that is to hold the files bit for bit (the developer, 2026-11-17: "Bit for bit, per
computer"). ``tests/test_reference.py`` compares today's code against these files.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np

from tests.reference import cases


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=cases.ROOT, check=True, capture_output=True,
                          text=True).stdout.rstrip("\n")


def _code_sha256() -> str:
    """One fingerprint over every Python file in ``cases.CODE``, in path order: it names the code
    the numbers came from even when the commit does not."""
    digest = hashlib.sha256()
    for entry in cases.CODE:
        for f in sorted((cases.ROOT / entry).rglob("*.py")):
            digest.update(str(f.relative_to(cases.ROOT)).encode() + b"\0" + f.read_bytes())
    return digest.hexdigest()


def _same(path, arrays: dict[str, np.ndarray]) -> bool:
    """Whether the file at ``path`` already holds exactly these arrays (np.savez stamps the time
    into the file, so writing it again would change its bytes with no number moved)."""
    if not path.exists():
        return False
    with np.load(path) as held:
        return sorted(held.files) == sorted(arrays) and all(
            held[k].dtype == v.dtype and held[k].shape == v.shape
            and held[k].tobytes() == v.tobytes() for k, v in arrays.items())


def main() -> None:
    # read before any file is written: the files this writes are the repository's too
    dirty = bool(_git("status", "--porcelain"))
    # a case taken out of the registry takes its frozen file with it
    for path in cases.REFERENCE_DIR.glob("*.npz"):
        if path.stem not in cases.CASES:
            path.unlink()
            print(f"removed {path.name}")
    for name in cases.CASES:
        path, arrays = cases.REFERENCE_DIR / f"{name}.npz", cases.produce(name)
        if _same(path, arrays):
            print(f"kept {name}.npz: no array moved")
        else:
            np.savez(path, **arrays)
            print(f"wrote {name}.npz")
    manifest = {
        "made": {"git_sha": _git("rev-parse", "--short", "HEAD"),
                 "git_dirty": dirty,
                 "code_changed": _git("status", "--porcelain", "--", *cases.CODE).splitlines(),
                 "code_sha256": _code_sha256(),
                 "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "computer": cases.computer()},
        "left_out": cases.LEFT_OUT,
        "cases": {name: cases.conditions_of(name) for name in cases.CASES},
    }
    cases.MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote manifest.json")


if __name__ == "__main__":
    main()
