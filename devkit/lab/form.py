"""The run view's form as plain data -- what each control offers and says, which settings a gate
shuts, the sweep row, and the ``deepc run`` the form launches. No Panel here.

Everything is read off :func:`record.config.fields_of`; the form restates no rule the Config
holds. A field back at its declared default is **unset** (an empty box runs as the default);
only set fields become ``--set`` words. A gate (``needs``) is read against the typed value of
the setting it names, or against the sweep rungs when that setting is the sweep axis.
"""

from __future__ import annotations

import json
from pathlib import PurePath
from typing import Any, Optional

INF = "∞"
ELLIPSIS = "…"
#: What an empty box, or a pick's first option, says for a setting whose default is the code's
#: own (``code_default``: ``None`` in the Config, resolved by the run and saved in its record's
#: constants): an empty placeholder said nothing about what the run would use.
CODE_DEFAULT = "Code default"


def as_text(value: Any) -> str:
    """A value as the text ``--set`` reads back: ``true``/``false``, an integral float without
    its point, a list as JSON (a list of expressions holds quotes ``str`` would write as Python's),
    else ``str``."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return str(value)


def hover(f: dict[str, Any]) -> str:
    """The field's doc, then the bounds it declares as an interval: ``[`` for an included end,
    ``(`` for an excluded one, ``∞`` for none. Declared, not described."""
    doc = f.get("doc") or ""
    if f.get("low") is None and f.get("high") is None:
        return doc
    lo = f"(-{INF}" if f["low"] is None else ("(" if f["low_exclusive"] else "[") + as_text(f["low"])
    hi = f"{INF})" if f["high"] is None else as_text(f["high"]) + (")" if f["high_exclusive"] else "]")
    return f"{doc} {lo} {ELLIPSIS} {hi}".strip()


def term(f: dict[str, Any]) -> str:
    """``<label> [unit]``, the words a control wears."""
    return f"{f['label']} [{f['unit']}]" if f.get("unit") else f["label"]


class Form:
    """One piece's form: the typed values, the sweep row and the record name, as plain data."""

    def __init__(self, piece: dict[str, Any]) -> None:
        self.piece = piece
        self.fields: list[dict[str, Any]] = list(piece["fields"])
        self.by_name = {f["name"]: f for f in self.fields}
        self.set: dict[str, str] = {}
        self.name = ""
        self.sweep_field = ""
        self.sweep_values = ""
        #: What the last edit took back from behind a gate that shut. Said in :meth:`note`,
        #: because a typed value that vanishes without a word is a value the run ignores.
        self.taken: list[str] = []
        #: Each piece's saved runs that can be read, as paths: what a setting naming one
        #: (``record_of``) offers. The page fills it from the folder it reads.
        self.saved: dict[str, list[str]] = {}

    # -- words
    def label(self, name: str) -> str:
        return self.by_name[name]["label"]

    def effective(self, name: str) -> str:
        """The text the run would use: typed, else the declared default."""
        return self.set.get(name, as_text(self.by_name[name]["default"]))

    def shown_default(self, f: dict[str, Any]) -> str:
        """What an empty box runs as -- the placeholder: the declared default, or
        :data:`CODE_DEFAULT` for one the run resolves."""
        return CODE_DEFAULT if f.get("code_default") else as_text(f["default"])

    def options(self, f: dict[str, Any]) -> Optional[list[dict[str, str]]]:
        """What a pick list offers: ``{value, words, line}`` per option, the empty value first
        worded as the default; ``None`` for a typed box. A setting naming a saved run offers
        that piece's runs, each worded by its name, the path its line -- and the path it holds,
        when no saved run is that path, so a run's own is never swapped for another."""
        lines = f.get("lines") or {}
        default = as_text(f["default"])
        if f.get("record_of"):
            paths = [p for p in self.saved.get(f["record_of"], []) if p != default]
            held = self.set.get(f["name"], "")
            paths += [held] if held and held not in paths else []
            return ([{"value": "", "words": f"{PurePath(default).stem} (default)",
                      "line": default}]
                    + [{"value": p, "words": PurePath(p).stem, "line": p} for p in paths])
        if f.get("choices"):
            values = [as_text(c) for c in f["choices"]]
        elif f["type"] == "bool":
            values = ["true", "false"]
        else:
            return None
        first = CODE_DEFAULT if f.get("code_default") else f"{default} (default)"
        out = [{"value": "", "words": first, "line": lines.get(default, "")}]
        out += [{"value": v, "words": v, "line": lines.get(v, "")} for v in values if v != default]
        return out

    # -- the sweep row
    def rungs(self) -> list[str]:
        """The sweep values as the terminal splits them (on commas); ``[]`` while empty. Read to
        decide what a gate sees; the run itself is launched with the typed text."""
        return [v.strip() for v in self.sweep_values.split(",") if v.strip()]

    def swept(self) -> str:
        """The swept field: named AND given something. Whether what was given is an acceptable
        list of values is the terminal's answer, in its own words, not this module's."""
        return self.sweep_field if self.sweep_field and self.sweep_values.strip() else ""

    def axes(self) -> list[tuple[str, str]]:
        """What the sweep row can vary: every setting but a list of numbers, whose own commas
        the terminal would split as the sweep's, and but a saved run's path, which the name of
        the record each rung writes cannot hold."""
        return [(f["name"], term(f)) for f in self.fields
                if f["type"] != "list" and not f.get("record_of")]

    # -- gates
    @staticmethod
    def holds(gate: dict[str, Any], value: str) -> bool:
        return value in [as_text(v) for v in gate.get("in", [])]

    def gate_status(self, other: str, gate: dict[str, Any]) -> str:
        """``all`` / ``none`` / ``some``: whether the setting ``other`` opens ``gate`` -- read
        against the sweep rungs when ``other`` is the sweep axis."""
        if other == self.swept():
            opens = [self.holds(gate, v) for v in self.rungs()]
            return "all" if all(opens) else "none" if not any(opens) else "some"
        return "all" if self.holds(gate, self.effective(other)) else "none"

    def state_of(self, f: dict[str, Any]) -> tuple[bool, str]:
        """``(off, hover)``: greyed behind a closed gate or as the sweep axis, with the reason;
        a gate only some rungs open is not greyed, it warns."""
        if f["name"] == self.swept():
            return True, f"{f['label']} is the sweep axis; its values come from the sweep row"
        words = hover(f)
        for other, gate in (f.get("needs") or {}).items():
            status = self.gate_status(other, gate)
            wanted = " or ".join(as_text(v) for v in gate.get("in", []))
            if status == "none":
                return True, (f"{f['label']} applies only when {self.label(other)} is {wanted}; "
                              f"it is {self.effective(other)}")
            if status == "some":
                words += (f" — only some sweep values of {self.label(other)} open it "
                          f"({wanted}); the run stops at the first that does not")
        return False, words

    # -- typing
    def type_or_pick(self, name: str, text: str) -> list[str]:
        """Set (or clear, back at the default or empty) one field, then take back every set
        field left behind a gate that shut; returns the names taken back."""
        text = (text or "").strip()
        if not text or text == as_text(self.by_name[name]["default"]):
            self.set.pop(name, None)
        else:
            self.set[name] = text
        return self._shut_back()

    def sweep(self, field_name: str, values: str) -> list[str]:
        """Set the sweep row; the axis's own typed value is taken back (a setting is swept once,
        and not also set)."""
        self.sweep_field, self.sweep_values = field_name or "", values or ""
        return self._shut_back()

    def _shut_back(self) -> list[str]:
        """Drop every set field now behind a shut gate, or now the sweep axis. What went is kept
        in :attr:`taken` so :meth:`note` says so: the alternative is a value the run would
        silently ignore, which is the thing ``record.config.gated`` refuses outright."""
        taken: list[str] = []
        for name in list(self.set):
            if self.state_of(self.by_name[name])[0]:
                del self.set[name]
                taken.append(name)
        self.taken = taken
        return taken

    def note(self) -> tuple[str, bool]:
        """``(words, warns)``: how many settings are set, what the last edit took back, and the
        warnings the terminal itself would act on."""
        words = f"{len(self.set)} set"
        warnings = []
        if self.taken:
            them = "them" if len(self.taken) > 1 else "it"
            warnings.append(", ".join(self.label(n) for n in self.taken)
                            + f" taken back: the run would ignore {them}")
        rungs = self.rungs()
        if len(set(rungs)) != len(rungs):
            warnings.append("the sweep repeats a value — the terminal refuses the run")
        for f in self.fields:
            for other, gate in (f.get("needs") or {}).items():
                if other == self.swept() and self.gate_status(other, gate) == "some" \
                        and f["name"] in self.set:
                    warnings.append(f"{f['label']} is set, and only some sweep values open it")
        if warnings:
            return f"{words} — " + "; ".join(warnings), True
        return words, False

    # -- names
    def would_write(self, name: str) -> list[str]:
        """The record names this form would write under ``name``: one per sweep rung, named as
        the terminal names them (``<name>-<setting>-<value>``), else ``name`` alone."""
        swept = self.swept()
        if not swept:
            return [name]
        return [f"{name}-{swept}-{v}" for v in self.rungs()] or [name]

    def _held(self, runs: list[dict[str, Any]]) -> set[str]:
        return {r["name"] for r in runs if r["piece"] == self.piece["name"]}

    def free_name(self, runs: list[dict[str, Any]]) -> str:
        """The first ``<piece>-N`` no record uses, a sweep's suffixed names included: a held
        ``<piece>-1-Ts-0.1`` means ``<piece>-1`` is the name of a sweep already run."""
        held = self._held(runs)
        stem = self.piece["name"]
        n = 1
        while any(h == f"{stem}-{n}" or h.startswith(f"{stem}-{n}-") for h in held):
            n += 1
        return f"{stem}-{n}"

    def name_note(self, runs: list[dict[str, Any]]) -> str:
        """Which records this run would overwrite, by name -- under the name in the box, or the
        one the placeholder offers when it is empty."""
        name = self.name.strip() or self.free_name(runs)
        held = self._held(runs)
        over = [n for n in self.would_write(name) if n in held]
        if not over:
            return ""
        many = len(over) > 1
        return (f"{', '.join(over)} exist{'' if many else 's'} — "
                f"Run overwrites {'them' if many else 'it'}")

    # -- from a run, and back to nothing
    def use(self, config: dict[str, Any]) -> tuple[int, list[str]]:
        """Load a saved config's values away from the defaults. Returns ``(landed, dropped)``:
        what the form now holds, and the keys this Config no longer declares. The gates checked
        are today's, not the saved run's -- a record older than the settings it ran under is
        exactly what ``state._stale`` marks."""
        self.set = {}
        self.name = ""
        self.sweep_field = self.sweep_values = ""
        dropped = []
        for key, value in config.items():
            f = self.by_name.get(key)
            if f is None:
                dropped.append(key)
            elif as_text(value) != as_text(f["default"]):
                self.set[key] = as_text(value)
        self._shut_back()
        return len(self.set), dropped

    def clear(self) -> None:
        self.set = {}
        self.name = ""
        self.sweep_field = self.sweep_values = ""
        self.taken = []

    # -- the command
    def launch_words(self) -> tuple[list[str], list[str]]:
        """``(overrides, sweeps)`` as ``--set`` and ``--sweep`` take them; the swept field only
        in ``sweeps``, carrying the values as they were typed -- what is an acceptable list is
        the terminal's rule, and the terminal says so itself."""
        swept = self.swept()
        overrides = [f"{k}={v}" for k, v in self.set.items() if k != swept]
        sweeps = [f"{swept}={self.sweep_values.strip()}"] if swept else []
        return overrides, sweeps
