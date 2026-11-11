"""Settings: a piece's Config, each setting a ``param`` parameter. Adapted from BESS's
``devkit/record/config.py`` (itself RCWS's), trimmed to what this repository's pieces use.

param holds the type, the default, the ``label`` (the words a person reads), the ``doc`` (one
line: what it is, what changing it does), the ``bounds`` (declared only where the code enforces
one) and the ``objects`` of a choice, and refuses a wrong type, a value outside a bound or a pick
outside the choices in its own words. What param has no slot for rides in ``metadata``:

- ``group`` (required): the heading a form puts the setting under;
- ``unit``: the unit a person types it in;
- ``needs``: what switches the setting on -- ``{other_setting: {"in": [named values]}}``. A value
  set behind a closed gate is refused on construction (:func:`gated`), because the run would
  ignore it and the saved config would claim it;
- ``replicate``: ``True`` for a setting that only re-draws the noise (a seed);
- ``record_of``: on a path setting, the piece whose saved run it names.

A setting whose default is ``None`` means "the plant's own": the piece resolves it when it runs
and saves what it resolved. A list setting holds numbers (the Q and R weights): JSON or commas
on the command line. ``none`` on the command line sets a setting that allows it back to ``None``.

A built Config is immutable; a changed one is a new Config (:func:`with_overrides`).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import param

__all__ = ["Settings", "fields_of", "gated", "with_overrides", "to_dict", "from_dict",
           "coerce_value", "parse_override"]


class Settings(param.Parameterized):
    """Base of every piece's Config. See the module."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name, value in vars(cls).items():
            if isinstance(value, param.Parameter):
                if not (value.metadata or {}).get("group"):
                    raise TypeError(f"{cls.__name__}.{name} declares no group; a form has to put "
                                    f"a setting under one (metadata={{'group': ...}})")
                value.constant = True

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)
        gated(self)


def _parameters(cls: type) -> dict[str, param.Parameter]:
    """The declared parameters, in declaration order, without param's own ``name``."""
    return {n: p for n, p in cls.param.objects(instance=False).items() if n != "name"}


def _gate_holds(gate: Mapping[str, Any], value: Any) -> bool:
    if set(gate) != {"in"}:
        raise ValueError(f"a gate is {{'in': [named values]}}, got {gate!r}")
    return value in gate["in"]


def gated(cfg: Settings) -> None:
    """Refuse a setting set behind a closed gate; see the module."""
    for name, p in _parameters(type(cfg)).items():
        needs = (p.metadata or {}).get("needs")
        if not needs:
            continue
        closed = [(other, gate) for other, gate in needs.items()
                  if not _gate_holds(gate, getattr(cfg, other))]
        if closed and getattr(cfg, name) != p.default:
            gates = "; ".join(f"{other} is one of {gate['in']!r} (it is {getattr(cfg, other)!r})"
                              for other, gate in closed)
            raise ValueError(f"{name}={getattr(cfg, name)!r} is set, but {name} applies only "
                             f"when {gates}; the run would ignore it")


def _type_text(p: param.Parameter) -> str:
    return {param.Integer: "int", param.Number: "float", param.Boolean: "bool",
            param.String: "str", param.Selector: "choice", param.List: "list"}.get(
        type(p), type(p).__name__)


def fields_of(cls: type) -> list[dict[str, Any]]:
    """One dict per setting -- the one reader every surface (terminal, tests, a form) uses."""
    out = []
    for name, p in _parameters(cls).items():
        meta = p.metadata or {}
        numeric = isinstance(p, param.Number)
        bounds = (getattr(p, "bounds", None) if numeric else None) or (None, None)
        out.append({
            "name": name, "type": _type_text(p), "default": p.default,
            "label": p.label or name, "unit": meta.get("unit", ""), "group": meta["group"],
            "doc": (p.doc or "").strip(),
            "choices": list(p.objects) if isinstance(p, param.Selector) else None,
            "low": bounds[0], "high": bounds[1], "code_default": p.default is None,
            "needs": meta.get("needs"), "replicate": bool(meta.get("replicate")),
            "record_of": meta.get("record_of", ""),
        })
    return out


_TRUE, _FALSE = {"true", "1", "yes", "on"}, {"false", "0", "no", "off"}


def coerce_value(p: param.Parameter, value: Any) -> Any:
    """A command-line text as the setting's own type; a typed value passes through, but a list
    setting's numbers become floats (a config file may write the weight 10 as an integer)."""
    if isinstance(p, param.List) and isinstance(value, (list, tuple)):
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            raise ValueError(f"{p.name}: {value!r} is not a list of numbers") from None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() == "none" and p.allow_None:
        return None
    if isinstance(p, param.Boolean):
        if text.lower() in _TRUE:
            return True
        if text.lower() in _FALSE:
            return False
        raise ValueError(f"{p.name}: a bool is one of {sorted(_TRUE | _FALSE)}, not {text!r}")
    try:
        if isinstance(p, param.Integer):
            return int(text)
        if isinstance(p, param.Number):
            return float(text)
        if isinstance(p, param.List):
            held = json.loads(text) if text.startswith("[") else [float(v) for v in text.split(",")]
            if not isinstance(held, list):
                raise ValueError
            return [float(v) for v in held]
    except (ValueError, TypeError, json.JSONDecodeError):
        kind = {param.Integer: "an integer", param.Number: "a number"}.get(
            type(p), "a list of numbers, [1, 2] or 1,2")
        raise ValueError(f"{p.name}: {text!r} is not {kind}") from None
    if isinstance(p, param.Selector):
        by_text = {str(o): o for o in p.objects}
        if text not in by_text:
            raise ValueError(f"{p.name}: {text!r} is not one of {sorted(by_text)}")
        return by_text[text]
    return text


def to_dict(cfg: Settings) -> dict[str, Any]:
    """The settings, in declaration order, as plain values."""
    return {n: getattr(cfg, n) for n in _parameters(type(cfg))}


def from_dict(cls: type, raw: Mapping[str, Any], where: str = "") -> Settings:
    """A Config from a mapping; an unknown key is refused, never dropped."""
    params = _parameters(cls)
    unknown = sorted(set(raw) - set(params))
    if unknown:
        raise ValueError(f"unknown setting(s) {unknown}{' in ' + where if where else ''} for "
                         f"{cls.__name__}; valid: {sorted(params)}")
    return cls(**{k: coerce_value(params[k], v) for k, v in raw.items()})


def with_overrides(cfg: Settings, pairs: Sequence[tuple[str, Any]]) -> Settings:
    """``cfg`` with each ``(setting, value)`` applied, rebuilt ONCE so a gate is checked on the
    final state, not on an intermediate one."""
    cls = type(cfg)
    params = _parameters(cls)
    values = to_dict(cfg)
    for name, value in pairs:
        if name not in params:
            raise ValueError(f"unknown setting {name!r} on {cls.__name__}; valid: {sorted(params)}")
        values[name] = coerce_value(params[name], value)
    return cls(**values)


def parse_override(item: str) -> tuple[str, str]:
    """``"setting=value"`` -> ``("setting", "value")``."""
    name, sep, value = item.partition("=")
    if not sep or not name:
        raise ValueError(f"expected SETTING=VALUE, got {item!r}")
    return name.strip(), value
