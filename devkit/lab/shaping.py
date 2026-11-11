"""The ways a Lab card shapes what it draws, each declared once: what it holds on the card, its
words, its kind, its group, what it offers, why a card cannot use it (or one of its options) right
now, and the keyword it hands the figure layer. A card's controls are these, in this order
(:data:`SHAPINGS`); its chart is drawn from what they hold (:func:`draw_keywords`), and a new way
of shaping a chart is one entry here (RCWS ``feat/unequal-state-mixing``, ``lab/shaping.py``).

Three, after the developer's cut of 2026-11-11: a chart card's runs (Overlay | Split | Mean |
RMS); a number card's setting on the bottom axis (X) and its runs (All | Mean | RMS). A control a
card cannot use is greyed with the reason on its hover: the figure layer's refusal in its own
words where it is one (``figure.chart.pool_refusal``).

No Panel here, so what a card offers is tested without a page.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Card", "Context", "Shaping", "SHAPINGS", "offered", "draw_keywords"]


@dataclass
class Card:
    """One open card: a ``chart`` (a panel the records declare) or a ``numbers`` card (a summary
    number against a setting, ``x``). ``runs`` and ``x`` are ``""`` for the card's default."""
    id: int
    kind: str
    chart: str = ""
    number: str = ""
    x: str = ""
    runs: str = ""


@dataclass(frozen=True)
class Context:
    """What a card's controls are read off: its kind and how many runs it draws; on a chart
    card why each statistic cannot pool its runs (``{"mean": why, "rms": why}``, ``""`` where it
    can); on a numbers card the settings on offer for its bottom axis (words -> setting) and
    whether a colour holds two runs at one value."""
    kind: str
    runs: int = 1
    axes: dict[str, str] = field(default_factory=dict)
    pool: dict[str, str] = field(default_factory=dict)
    pools: bool = False


def _fine(card: Card, ctx: Context) -> str:
    return ""


@dataclass(frozen=True)
class Shaping:
    """One way of shaping a card. ``kind``: ``list`` (a drop-down) or ``one`` (one of a few,
    side by side, an option greyed alone where it cannot be picked). ``options(card, ctx)`` are
    ``(words, value, why greyed)``; ``shown`` what it holds for a card; ``keywords`` what it hands
    the figure layer; ``unusable`` why the whole control is greyed, ``""`` when it is not;
    ``kinds`` the cards it stands on."""
    field: str
    words: str
    kind: str
    group: int
    shown: Callable[[Card, Context], Any]
    keywords: Callable[[Card, Context], dict[str, Any]]
    options: Callable[[Card, Context], list[tuple[str, Any, str]]] = lambda card, ctx: []
    kinds: tuple[str, ...] = ("chart", "numbers")
    unusable: Callable[[Card, Context], str] = _fine


# -- the chart card's runs: one plot, a row each, or pooled where the chart declares a statistic

def _chart_runs(card: Card, ctx: Context) -> str:
    value = card.runs or "one"
    return "one" if value in ("mean", "rms") and ctx.pool.get(value) else value


def _chart_runs_options(card: Card, ctx: Context) -> list[tuple[str, Any, str]]:
    return [("Overlay", "one", ""), ("Split", "rows", ""),
            ("Mean", "mean", ctx.pool.get("mean", "")), ("RMS", "rms", ctx.pool.get("rms", ""))]


def _one_run(card: Card, ctx: Context) -> str:
    return "One run shown" if ctx.runs < 2 else ""


# -- the numbers card's runs: each a point, or a colour's runs at one value pooled

def _numbers_runs(card: Card, ctx: Context) -> str:
    return (card.runs or "each") if ctx.pools else "each"


#: Every way of shaping a card, in the order a card shows them.
SHAPINGS: tuple[Shaping, ...] = (
    Shaping("x", "X", "list", 1, kinds=("numbers",),
            options=lambda card, ctx: [(w, v, "") for w, v in ctx.axes.items()],
            unusable=lambda card, ctx: "" if ctx.axes else "Nothing differs between the shown runs",
            shown=lambda card, ctx: card.x,
            keywords=lambda card, ctx: {"vs": card.x}),
    Shaping("runs", "Runs", "one", 2, kinds=("chart",), options=_chart_runs_options,
            unusable=_one_run, shown=_chart_runs,
            keywords=lambda card, ctx: {"runs": _chart_runs(card, ctx)}),
    Shaping("runs", "Runs", "one", 2, kinds=("numbers",),
            options=lambda card, ctx: [("All", "each", ""), ("Mean", "mean", ""),
                                       ("RMS", "rms", "")],
            unusable=lambda card, ctx: "" if ctx.pools else "No colour holds two runs at one value",
            shown=_numbers_runs,
            keywords=lambda card, ctx: {"runs": _numbers_runs(card, ctx)}),
)


def offered(card: Card, ctx: Context) -> list[tuple[Shaping, str]]:
    """``(shaping, why it is greyed or "")`` for every way the card shapes what it draws, in
    order -- every one, on every card of its kind."""
    return [(s, s.unusable(card, ctx)) for s in SHAPINGS if ctx.kind in s.kinds]


def draw_keywords(card: Card, ctx: Context) -> dict[str, Any]:
    """What the card draws, as the figure layer's keywords (``figure.chart.chart_of`` for a
    chart card, ``sweep_of`` for a numbers card)."""
    out: dict[str, Any] = {}
    for s, _ in offered(card, ctx):
        out.update(s.keywords(card, ctx))
    return out
