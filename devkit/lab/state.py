"""What the page lists, as plain data; every value is read, none computed.

Two sources only: the declarations (the registry and :func:`record.config.fields_of`) and the
files under the terminal's out-dir. A record is LISTED as the JSON it is -- its header, config,
provenance and numbers, never the traces -- and a record that is SHOWN is loaded through its
own class (:func:`load`), every rule of that loading the class's own, so the figure layer's
twin can draw what it declares.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from harness import NAMES, get_component
from record import (COMPONENT_KEY, HEADER_KEYS, NUMBER_SPECS_KEY, PANEL_GROUPS_KEY, PANELS_KEY,
                    read_header)
from record.config import fields_of

_HEADER = HEADER_KEYS
#: The keys every listed record carries, and nothing else.
RECORD_KEYS = ("piece", "name", "path", "mtime", "panels", "panel_groups", "number_specs",
               "numbers", "config", "provenance", "stale")
_CACHE: dict[str, tuple[tuple[float, int], dict[str, Any]]] = {}


def pieces() -> list[dict[str, Any]]:
    """Every registered piece: its name, its own words, the command that writes a record, and
    its settings as :func:`fields_of` declares them."""
    out = []
    for name in NAMES:
        component = get_component(name)
        out.append({"name": name, "label": component.Record.label or name,
                    "produced_by": component.Record.produced_by,
                    "fields": fields_of(component.Config)})
    return out


def records(root: str, out_dir: str) -> list[dict[str, Any]]:
    """Every ``<root>/<out_dir>/<piece>/*.json``, newest first. An absent folder is an empty
    list; a file that is not a record is listed carrying the refusal as its ``stale`` words."""
    rows = []
    for name in NAMES:
        folder = Path(root) / out_dir / name
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.suffix == ".json" and path.is_file():
                rows.append(_record(path, name))
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def _record(path: Path, piece: str) -> dict[str, Any]:
    st = path.stat()
    key = (st.st_mtime, st.st_size)
    hit = _CACHE.get(str(path))
    if hit and hit[0] == key:
        return hit[1]
    row = _read(path, piece, st.st_mtime)
    _CACHE[str(path)] = (key, row)
    return row


def _read(path: Path, piece: str, mtime: float) -> dict[str, Any]:
    """One row. A JSON that is not a record is LISTED, carrying ``read_header``'s words as its
    staleness, never raised: the out-dir is a folder people keep things in (a saved settings
    file is how a setup is kept, and ``deepc draw`` can be pointed at it), and one such file must
    not take the page down with it."""
    row = {"piece": piece, "name": path.stem, "path": str(path), "mtime": mtime,
           "panels": [], "panel_groups": {}, "number_specs": [], "numbers": {}, "config": {},
           "provenance": {}, "stale": ""}
    try:
        header = read_header(path)                    # the authority on what a record is
    except ValueError as exc:
        row["stale"] = str(exc)
        return row
    with open(path) as fh:
        raw = json.load(fh)
    panels, panel_words = _panels(header.get(PANELS_KEY))
    specs, spec_words = _number_specs(header.get(NUMBER_SPECS_KEY))
    row.update({
        "name": path.stem,                 # the file is the identity; `deepc draw` takes the path
        "panels": panels,
        "panel_groups": _groups(header.get(PANEL_GROUPS_KEY), panels),
        "number_specs": specs,
        "numbers": dict(raw.get("numbers", {})),
        "config": dict(raw.get("config", {})),
        "provenance": dict(raw.get("provenance", {})),
        "stale": "; ".join(w for w in (_stale(piece, raw), panel_words, spec_words) if w),
    })
    return row


def _panels(held: Any) -> tuple[list[str], str]:
    """The panel names, or none of them and the words why: whatever a page reads here it will
    subscript, so a header of another shape is refused ONCE, here, not where it is used."""
    if held is None:
        return [], ""
    if not isinstance(held, list) or not all(isinstance(p, str) for p in held):
        return [], f"its {PANELS_KEY!r} header is not a list of names"
    return list(held), ""


def _groups(held: Any, panels: list[str]) -> dict[str, str]:
    """Each panel's group as the header holds it; none for a file written before groups were
    declared, or one whose groups are not names, so that it lists ungrouped rather than fails."""
    if not isinstance(held, dict) or not all(isinstance(v, str) for v in held.values()):
        return {}
    return {p: held[p] for p in panels if p in held}


def _number_specs(held: Any) -> tuple[list[dict[str, Any]], str]:
    """The number declarations, each with at least the name and the words to show it by."""
    if held is None:
        return [], ""
    if not isinstance(held, list):
        return [], f"its {NUMBER_SPECS_KEY!r} header is not a list"
    out = []
    for spec in held:
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str) \
                or not isinstance(spec.get("label"), str):
            return [], f"its {NUMBER_SPECS_KEY!r} header does not declare a name and a label"
        out.append(spec)
    return out, ""


def _stale(piece: str, raw: dict[str, Any]) -> str:
    """Words for a record whose FIELD LAYOUT the piece has moved past, else an empty string.

    The field names only. ``Record.load`` also refuses a record whose saved values no longer
    fit the Config (a bound narrowed, a choice withdrawn, a gate added) and one whose body no
    longer decodes; re-stating those rules here would be a second implementation of them, so
    they are seen where they are enforced -- when the record is loaded, in its own words.
    """
    held_piece = raw.get(COMPONENT_KEY)
    if held_piece != piece:
        return f"is a {held_piece!r} record in the {piece} folder"
    declared = {f.name for f in dataclasses.fields(get_component(piece).Record)}
    held = set(raw) - set(_HEADER)
    missing, gone = sorted(declared - held), sorted(held - declared)
    parts = []
    if missing:
        parts.append(f"predates the field{'s' if len(missing) > 1 else ''} {', '.join(missing)}")
    if gone:
        parts.append(f"holds gone field{'s' if len(gone) > 1 else ''} {', '.join(gone)}")
    return "; ".join(parts)


def load(run: dict[str, Any]) -> Any:
    """The record behind a listed row, through its own class: READ off disk, with every rule of
    the reading the class's own (a body that no longer decodes, a value the Config no longer
    admits, are refused in its words). What the page hands to the figure layer's twin, and
    where it reads a record's constants."""
    return get_component(run["piece"]).Record.load(run["path"])


def under(path: str, root: str) -> bool:
    """Whether ``path`` really resolves inside ``root`` (symlinks and ``..`` included)."""
    real_root = os.path.realpath(root)
    return os.path.realpath(path).startswith(real_root.rstrip(os.sep) + os.sep)


def wrote(lines: list[str], root: str, out_dir: str) -> list[str]:
    """The printed lines that are existing files under the out-dir. The terminal prints one path
    per artefact it writes, relative to the root, so those lines are its output."""
    base = os.path.join(root, out_dir)
    out = []
    for line in lines:
        candidate = os.path.join(root, line.strip())
        if line.strip() and os.path.isfile(candidate) and under(candidate, base):
            out.append(candidate)
    return out
