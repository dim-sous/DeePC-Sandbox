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
from typing import Any, Optional, Sequence

from harness import NAMES, get_component
from record.config import fields_of
from terminal import DEFAULT_OUT_DIR, DRAW_SHAPING

#: A record's name: letters, digits, '.', '_' or '-', starting with a letter or digit, so it
#: can never read as a flag, a path or a second word.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
#: A folder of stills: a record's name, or names joined by '+' (a draw's default folder,
#: ``terminal.folder_of``) -- so '+' is allowed here and nowhere else (RCWS ``lab/jobs.py``).
_FOLDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\Z")

#: One command per process, whether launched or waited on. Each browser session builds its own
#: page, so two tabs of one server would otherwise run two `uv run` at once -- which race the
#: venv and, under the same record name, interleave two JSON streams into one file.
_LAUNCHED: "Optional[Job]" = None
_CAPTURING: "Optional[list[str]]" = None
_LAUNCHING = threading.Lock()


def _in_flight() -> Optional[str]:
    """The command this process is already running, as its line; ``None`` when it is free.
    Read under :data:`_LAUNCHING`."""
    if _LAUNCHED is not None and _LAUNCHED.running:
        return _LAUNCHED.line
    if _CAPTURING is not None:
        return line_of(_CAPTURING)
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


def draw_argv(piece: str, names: Sequence[str], out_dir: str = DEFAULT_OUT_DIR,
              root: Optional[str] = None, *, panels: Sequence[str] = (), number: str = "",
              table: bool = False, **keywords: Any) -> list[str]:
    """``uv run deepc draw <record> [<record> ...] [--panel P]* | --number N, the shaping flags
    [--table]``: the still of what a card draws, every flag read off the terminal's own
    declaration (``terminal.DRAW_SHAPING``) so the page can ask for nothing the command does not
    take. ``keywords`` are the figure layer's, as a card holds them (``lab.shaping``); a ``None``
    is what the chart declares and passes no flag.

    The record paths in the argv stay as given, because the command is run from ``root`` and has
    to read back as the line a person would type there; ``root`` is where this function looks
    for the records, which is not the directory the server happens to be running in. The draw
    writes where the terminal puts it by default: a folder beside the records, named after them.
    """
    piece = _piece(piece)
    if not names:
        raise ValueError("a draw needs at least one record")
    argv = ["uv", "run", "deepc", "draw"]
    for name in names:
        path = os.path.join(out_dir, piece, f"{_name(name)}.json")
        if not os.path.isfile(os.path.join(root, path) if root else path):
            raise ValueError(f"no record at {path}")
        argv.append(path)
    for panel in panels:
        argv += ["--panel", _name(panel)]
    if number:
        argv += ["--number", _name(number)]
    shaping = {k: (flag, kind) for k, flag, kind, *_ in DRAW_SHAPING}
    unknown = sorted(set(keywords) - set(shaping))
    if unknown:
        raise ValueError(f"{unknown} are not ways `deepc draw` shapes a drawing; it takes "
                         f"{sorted(shaping)}")
    for key, value in keywords.items():
        flag, kind = shaping[key]
        if value is None:
            continue
        if kind == "bool":
            argv.append(flag if value else "--no-" + flag[2:])
        elif kind == "flag":
            argv += [flag] if value else []
        elif kind == "range":
            # one word with '=': a range from below zero ('-5.0,10.0') read as its own word is
            # taken by argparse for an option, and the draw was refused
            argv.append(f"{flag}={float(value[0])!r},{float(value[1])!r}")
        elif kind == "names":
            argv += [flag, *(_name(v) for v in value)]
        else:
            argv += [flag, _name(str(value))]
    if table:
        argv.append("--table")
    return argv


def delete(piece: str, records: Sequence[str] = (), folders: Sequence[str] = (), *,
           root: str, out_dir: str = DEFAULT_OUT_DIR) -> list[str]:
    """Delete the named records (``<out_dir>/<piece>/<name>.json``) and folders of stills
    (``<out_dir>/<piece>/<name>/``, which hold only what a draw writes: ``.png``, ``.csv`` and
    their ``.json`` sidecars), and return what went, as paths under ``root``. Anything else is
    refused in words and nothing is deleted: a name that is not a record's, a path outside the
    out-dir, a folder holding a file no draw writes."""
    piece = _piece(piece)
    base = os.path.realpath(os.path.join(root, out_dir, piece))
    targets: list[tuple[str, str]] = []
    for name in records:
        path = os.path.join(base, f"{_name(name)}.json")
        if not os.path.isfile(path):
            raise ValueError(f"no record {name!r} under {out_dir}/{piece}/")
        targets.append(("file", path))
    for name in folders:
        if not _FOLDER.match(name or ""):
            raise ValueError(f"{name!r} is not a folder a draw writes: record names, or record "
                             f"names joined by '+'")
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            raise ValueError(f"no folder {name!r} under {out_dir}/{piece}/")
        odd = sorted(f for f in os.listdir(path) if not f.endswith(_DRAWN))
        if odd:
            raise ValueError(f"{out_dir}/{piece}/{name}/ holds {odd}, which no draw writes; "
                             f"it is left as it is")
        targets.append(("folder", path))
    for _, path in targets:
        if not os.path.realpath(path).startswith(base + os.sep):
            raise ValueError(f"{path} is not under {out_dir}/{piece}/")
    gone = []
    for kind, path in targets:
        if kind == "file":
            os.remove(path)
        else:
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
            os.rmdir(path)
        gone.append(os.path.relpath(path, root))
    return gone


#: What a draw writes into a folder: a still, a table, and each one's sidecar.
_DRAWN = (".png", ".csv", ".png.json", ".csv.json")


def capture(argv: Sequence[str], cwd: Optional[str] = None,
            timeout: float = 180.0) -> tuple[int, str]:
    """Run a command to completion and hand back ``(returncode, output)`` -- for a draw the page
    waits on. It CLAIMS the same one-at-a-time slot a run claims (:func:`_in_flight`): two
    ``uv run`` at once race the venv, so a second command, in either direction, is refused in
    words rather than queued. ``timeout`` is in seconds."""
    global _CAPTURING
    with _LAUNCHING:
        busy = _in_flight()
        if busy is not None:
            raise ValueError(f"a command is in progress: {busy}")
        _CAPTURING = list(argv)            # claimed here, so the next caller sees it
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        done = subprocess.run(list(argv), cwd=cwd, env=env, capture_output=True, text=True,
                              errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ValueError(f"{line_of(argv)} did not finish in {timeout:g} s") from None
    finally:
        with _LAUNCHING:
            _CAPTURING = None
    return done.returncode, (done.stdout or "") + (done.stderr or "")


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
