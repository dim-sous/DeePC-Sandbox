"""What the rail SHOWS, as plain data -- its rows and the words each wears. No Panel here, and
nothing computed: a value is shown as the record holds it."""

from __future__ import annotations

from typing import Any

from .form import as_text, hover, term


def text(value: Any) -> str:
    """A saved value as words: an em dash for nothing, four significant digits, lists joined."""
    if value is None:
        return "\u2014"
    if isinstance(value, bool):
        return as_text(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(text(v) for v in value)
    return str(value)


def when(run: dict[str, Any]) -> str:
    """``<date> · <sha>`` from the run's own provenance; empty when it carries none."""
    prov = run.get("provenance") or {}
    date = str(prov.get("date", ""))[:16].replace("T", " ")
    sha = str(prov.get("git_sha", ""))[:7]
    return " \u00b7 ".join(p for p in (date, sha) if p)


def run_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per run: its name, and why it cannot be read -- empty for a readable run. What sets
    a run apart is the compare table's to say, not the row's (RCWS ``view.run_rows``)."""
    return [{"name": r["name"], "stale": r["stale"]} for r in runs]


def matching(runs: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """The runs whose names hold every word of ``query``, each anywhere (``Ts 0.1``)."""
    words = query.lower().split()
    return [r for r in runs if all(w in r["name"].lower() for w in words)]


def varying(runs: list[dict[str, Any]], fields: list[dict[str, Any]]) -> list[str]:
    """The settings whose saved values differ across ``runs``; nothing differs under one run."""
    if len(runs) < 2:
        return []
    out = []
    for f in fields:
        if len({as_text(r["config"].get(f["name"])) for r in runs}) > 1:
            out.append(f["name"])
    return out


def _in_declared_order(runs: list[dict[str, Any]], key: str, name_of) -> list[str]:
    """The names under ``key`` across ``runs``, in the first run's own declared order, with any
    a later run adds appended: a record declares its order and the page does not re-sort it."""
    order: list[str] = []
    for run in runs:
        for item in run[key]:
            name = name_of(item)
            if name not in order:
                order.append(name)
    return order


def chart_rows(runs: list[dict[str, Any]], open_names: list[str]) -> list[dict[str, Any]]:
    """One row per panel the shown runs declare, in the first group a run declaring it puts it in
    (``""`` when none does), plus any chart already open that none of them declares -- an open
    card must always keep the control that closes it."""
    declared = _in_declared_order(runs, "panels", lambda p: p)
    rows = []
    for panel in declared + [p for p in open_names if p not in declared]:
        holders = [r for r in runs if panel in r["panels"]]
        if not holders:
            sub = "No shown run draws it"
        elif len(holders) == len(runs):
            sub = ""
        else:
            sub = f"{len(holders)} of {len(runs)} runs"
        # the first group a holder names: a run whose file predates groups names none, and
        # must not blank the chart's group for the runs that do (code review, 2026-10-27)
        group = next((g for r in holders if (g := (r.get("panel_groups") or {}).get(panel, ""))),
                     "")
        title = next((w for r in holders if (w := (r.get("panel_titles") or {}).get(panel, ""))),
                     panel)
        rows.append({"name": panel, "title": title, "open": panel in open_names, "sub": sub,
                     "group": group})
    return rows


def grouped(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """``rows`` under their groups, the groups in the order they first appear."""
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["group"], []).append(row)
    return list(out.items())


def _flat(held: Any, prefix: str = "") -> dict[str, Any]:
    """A nested dict of constants as one level, keyed ``group · key``."""
    out: dict[str, Any] = {}
    if not isinstance(held, dict):
        return out
    for key, value in held.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, f"{name} · "))
        else:
            out[name] = value
    return out


#: What the compare table shows beside the numbers: no settings, the ones that differ, or all.
SETTINGS = ("none", "differ", "all")


def compare_rows(runs: list[dict[str, Any]], fields: list[dict[str, Any]], *,
                 settings: str = "none", constants: list[Any] | None = None,
                 declared: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The shown runs column by column: the NUMBERS, in the groups their record declares
    (``Number.group``; ``"Numbers"`` for a file written before groups) and the headline ones
    first within a group; then, as ``settings`` asks (:data:`SETTINGS`), the settings group by
    group and the CONSTANTS a run used beyond its Config (``constants``: one dict per run, read
    off the loaded record, ``None`` for a run that carries none; nested dicts read
    ``group · key``). Each group opens with a ``head`` row of its own name.

    Every value is the one its record holds, worded by :func:`text`; a number a run does not
    carry is an em dash. A number row's ``key`` is its name, which opens its card.
    ``declared`` are the number declarations as the first run's class makes them now, which a
    page reads off the loaded record so that a file written before its numbers were grouped is
    compared in its class's groups; ``None`` reads each file's own header.

    Whether a row differs is decided on the SAVED values, never on the words they are shown by:
    0.00267124 and 0.00267139 both read ``0.002671`` at four digits, and a table that called
    those equal would certify a refactor that moved the numbers (CLAUDE.md ``[B4]``).
    """
    if settings not in SETTINGS:
        raise ValueError(f"settings is one of {SETTINGS}; {settings!r} given")
    held = ([{**run, "number_specs": declared} for run in runs[:1]] + runs[1:]
            if declared is not None and runs else runs)
    specs: dict[str, dict[str, Any]] = {}
    for run in held:
        for spec in run["number_specs"]:
            specs.setdefault(spec["name"], spec)
    declared_order = _in_declared_order(held, "number_specs", lambda s: s["name"])
    order = sorted(declared_order, key=lambda n: not specs[n].get("headline"))  # headline first
    groups = list(dict.fromkeys(specs[n].get("group") or "Numbers" for n in declared_order))

    rows: list[dict[str, Any]] = []

    def add(kind: str, group: str, key: str, words: str, hover_words: str, held: list[Any]) -> None:
        rows.append({"kind": kind, "group": group, "key": key, "term": words,
                     "hover": hover_words, "values": [text(v) for v in held],
                     "differs": len({as_text(v) for v in held}) > 1})

    for group in groups:
        in_group = [n for n in order if (specs[n].get("group") or "Numbers") == group]
        add("head", group, "", group, "", [])
        for name in in_group:
            spec = specs[name]
            unit = spec.get("unit") or ""
            add("number", group, name, f"{spec['label']} [{unit}]" if unit else spec["label"],
                spec.get("doc") or "", [r["numbers"].get(name) for r in runs])
    if settings == "none":
        return rows

    differ = set(varying(runs, fields))
    for group in dict.fromkeys(f["group"] for f in fields):
        in_group = [f for f in fields if f["group"] == group
                    and (settings == "all" or f["name"] in differ)]
        if not in_group:
            continue
        add("head", group, "", group, "", [])
        for f in in_group:
            add("setting", group, f["name"], term(f), hover(f),
                [r["config"].get(f["name"]) for r in runs])

    flat = [_flat(c) for c in (constants or [])]
    if any(flat):
        keys: list[str] = []
        for held in flat:
            keys.extend(k for k in held if k not in keys)
        kept = [k for k in keys
                if settings == "all" or len({as_text(h.get(k)) for h in flat}) > 1]
        if kept:
            add("head", "Constants", "", "Constants", "", [])
            for key in kept:
                add("constant", "Constants", key, key, "A value the run used beyond its "
                    "settings, read off the record", [h.get(key) for h in flat])
    return rows


def compare_count(rows: list[dict[str, Any]], runs: list[dict[str, Any]],
                  fields: list[dict[str, Any]]) -> str:
    """What the table holds: how many numbers under one run; under several, how many settings
    and numbers differ -- the settings counted whether or not the table shows them. Under one
    run nothing is being compared, so it does not say that nothing differs."""
    numbers = sum(1 for r in rows if r["kind"] == "number")
    if len(runs) < 2:
        return f"{numbers} number{'s' if numbers != 1 else ''}; tick another run to compare"
    settings = len(varying(runs, fields))
    moved = sum(1 for r in rows if r["kind"] == "number" and r["differs"])
    if not settings and not moved:
        return f"{numbers} number{'s' if numbers != 1 else ''}, none of them differs"
    return (f"{settings} setting{'s' if settings != 1 else ''} · "
            f"{moved} number{'s' if moved != 1 else ''} differ")
