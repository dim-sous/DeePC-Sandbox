"""Launching the terminal's commands -- the same line a person would type, in a subprocess.

Three consequences. The command is the typeable line and is printed as ``lab: <line>``; it is an
argv list, never a shell string, and every word in it is checked against a declaration this
module read (the registry, :func:`record.config.fields_of`); its output is the terminal's own,
refusals included -- the Lab restates no rule.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Optional, Sequence

from harness import NAMES, get_component
from record.config import fields_of
from terminal import DEFAULT_OUT_DIR

#: A record's name: letters, digits, '.', '_' or '-', starting with a letter or digit, so it
#: can never read as a flag, a path or a second word.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

#: One command per process. Each browser session builds its own
#: page, so two tabs of one server would otherwise run two `uv run` at once -- which race the
#: venv and, under the same record name, interleave two JSON streams into one file.
_LAUNCHED: "Optional[Job]" = None
_LAUNCHING = threading.Lock()


def _in_flight() -> Optional[str]:
    """The command this process is already running, as its line; ``None`` when it is free.
    Read under :data:`_LAUNCHING`."""
    if _LAUNCHED is not None and _LAUNCHED.running:
        return _LAUNCHED.line
    return None


def line_of(argv: Sequence[str]) -> str:
    """The argv as one typeable line; ``shlex.split`` of it gives the argv back."""
    return shlex.join(argv)


@dataclass
class Job:
    """One launched command: its argv, the lines it printed so far, and how it ended."""

    argv: list[str]
    output: list[str] = field(default_factory=list)
    returncode: Optional[int] = None
    process: Optional[subprocess.Popen] = None
    #: True once :meth:`stop` terminated it, so its exit code is not read as a refusal.
    stopped: bool = False

    @property
    def line(self) -> str:
        return line_of(self.argv)

    @property
    def running(self) -> bool:
        return self.process is not None and self.returncode is None

    def stop(self) -> bool:
        """Terminate a running job; False when it never started or already ended."""
        if not self.running:
            return False
        self.stopped = True
        self.process.terminate()
        return True


def _piece(piece: str) -> str:
    if piece not in NAMES:
        raise ValueError(f"unknown piece {piece!r}; registered: {list(NAMES)}")
    return piece


def _setting(item: str, piece: str, names: set[str]) -> str:
    key, sep, value = item.partition("=")
    if not sep or not key or not value:
        raise ValueError(f"{item!r} is not SETTING=VALUE")
    if key not in names:
        raise ValueError(f"{key!r} is not a setting of {piece}; its settings: {sorted(names)}")
    return item


def _name(name: str) -> str:
    if not _NAME.match(name or ""):
        raise ValueError(f"{name!r} cannot name a record: letters, digits, '.', '_' or '-', "
                         f"starting with a letter or digit")
    return name


def run_argv(piece: str, overrides: Sequence[str] = (), name: str = "",
             out_dir: str = DEFAULT_OUT_DIR, sweeps: Sequence[str] = ()) -> list[str]:
    """``uv run deepc run <piece> [--set K=V]* [--sweep K=V1,V2]* --name <name> [--out-dir D]``,
    ``--out-dir`` only when it is not the terminal's default. Every word checked."""
    piece = _piece(piece)
    names = {f["name"] for f in fields_of(get_component(piece).Config)}
    argv = ["uv", "run", "deepc", "run", piece]
    for item in overrides:
        argv += ["--set", _setting(item, piece, names)]
    for item in sweeps:
        argv += ["--sweep", _setting(item, piece, names)]
    argv += ["--name", _name(name)]
    if out_dir != DEFAULT_OUT_DIR:
        argv += ["--out-dir", out_dir]
    return argv


def delete(piece: str, records: Sequence[str] = (), *,
           root: str, out_dir: str = DEFAULT_OUT_DIR) -> list[str]:
    """Delete the named records (``<out_dir>/<piece>/<name>.json``) and return what went, as
    paths under ``root``. Anything else is refused in words and nothing is deleted: a name that
    is not a record's, a path outside the out-dir."""
    piece = _piece(piece)
    base = os.path.realpath(os.path.join(root, out_dir, piece))
    targets: list[str] = []
    for name in records:
        path = os.path.join(base, f"{_name(name)}.json")
        if not os.path.isfile(path):
            raise ValueError(f"no record {name!r} under {out_dir}/{piece}/")
        if not os.path.realpath(path).startswith(base + os.sep):
            raise ValueError(f"{path} is not under {out_dir}/{piece}/")
        targets.append(path)
    for path in targets:
        os.remove(path)
    return [os.path.relpath(path, root) for path in targets]


def launch(argv: Sequence[str], cwd: Optional[str] = None) -> Job:
    """Start the command; a daemon thread pumps its lines into ``job.output`` as they come.

    One command per process (:func:`_in_flight`): a second one is refused in words rather than
    run beside the first. ``errors="replace"`` and the ``finally`` are what keep a job from
    latching ``running`` forever -- whatever the child prints, the exit code is recorded.
    """
    global _LAUNCHED
    with _LAUNCHING:
        busy = _in_flight()
        if busy is not None:
            raise ValueError(f"a command is already in progress: {busy}")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen(list(argv), cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace",
                                   bufsize=1)
        job = Job(list(argv), process=process)
        _LAUNCHED = job

    def pump() -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                job.output.append(line)
        except Exception as exc:                       # the page must still learn it ended
            job.output.append(f"lab: reading the output failed: {exc!r}\n")
        finally:
            if process.stdout is not None:
                process.stdout.close()
            job.returncode = process.wait()

    threading.Thread(target=pump, daemon=True).start()
    return job
