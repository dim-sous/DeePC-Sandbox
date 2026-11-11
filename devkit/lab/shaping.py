"""The ways a Lab card shapes what it draws, each declared once: what it holds on the card, its
words, its kind, its group, what it offers, why a card cannot use it (or one of its options) right
now, and the keyword it hands the figure layer. A card's controls are these, in this order
(:data:`SHAPINGS`); its chart and its still are both drawn from what they hold
(:func:`draw_keywords`) -- one declaration, so a card and its still cannot disagree, and a new
way of shaping a chart is one entry here (RCWS ``feat/unequal-state-mixing``, ``lab/shaping.py``).

The controls stand in groups by what they act on -- the axes (``x``, ``log x``, ``log y``), the
runs (``colour by``, ``runs``), the drawing (``dots | lines``), the view (``chart | table``) --
every one on every card, and one a card cannot use greyed with the reason on its hover: a
figure-layer refusal in its own words where it is one (``figure.chart.pool_refusal``,
``figure.render.log_refusal``).

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
    number against a setting). ``log_y`` and ``points`` are ``None`` while they follow what the
    chart declares; ``runs`` and ``x`` are ``""`` for the card's default; ``by`` is ``None`` for
    every setting that tells the runs apart. ``saved`` is what its last save wrote."""
    id: int
    kind: str
    chart: str = ""
    number: str = ""
    x: str = ""
    log_x: bool = False
    log_y: bool | None = None
    colour_by: str = "run"
    by: tuple[str, ...] | None = None
    runs: str = ""
    points: bool | None = None
    table: bool = False
    saved: str = ""


@dataclass(frozen=True)
class Context:
    """What a card's controls are read off: its kind and how many runs it draws; the bottom
    axis's options (words -> value); why each statistic cannot pool its runs
    (``{"mean": why, "rms": why}``, ``""`` where it can); what the chart declares (its log
    ordinate, its dots); on a numbers card the settings that tell the runs apart (words ->
    setting) and whether a colour holds two runs at one value; and, once the chart is built, why
    it cannot take a logarithmic bottom axis or ordinate."""
    kind: str
    runs: int = 1
    axes: dict[str, str] = field(default_factory=dict)
    pool: dict[str, str] = field(default_factory=dict)
    declared_log_y: bool = False
    declared_points: bool = False
    telling: dict[str, str] = field(default_factory=dict)
    pools: bool = False
    log_x_refusal: str = ""
    log_y_refusal: str = ""


def _fine(card: Card, ctx: Context) -> str:
    return ""


@dataclass(frozen=True)
class Shaping:
    """One way of shaping a card. ``kind``: ``list`` (a drop-down), ``switch`` (on or off),
    ``one`` (one of a few, side by side, an option greyed alone where it cannot be picked) or
    ``several`` (any of a few). ``options(card, ctx)`` are ``(words, value, why greyed)``;
    ``shown`` what it holds for a card; ``keywords`` what it hands the figure layer;
    ``unusable`` why the whole control is greyed, ``""`` when it is not; ``kinds`` the cards it
    stands on."""
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


def _colour_unusable(card: Card, ctx: Context) -> str:
    if ctx.runs < 2:
        return "One run shown"
    runs = _chart_runs(card, ctx)
    if runs in ("mean", "rms"):
        return "The pooled runs are one line per quantity"
    if runs == "rows":
        return "Split: the colour tells the quantities apart"
    return ""


# -- the numbers card's runs: each a point, or a colour's runs at one value pooled

def _numbers_runs(card: Card, ctx: Context) -> str:
    return (card.runs or "each") if ctx.pools else "each"


def _log_x_unusable(card: Card, ctx: Context) -> str:
    return "" if card.log_x else ctx.log_x_refusal


def _log_y_shown(card: Card, ctx: Context) -> bool:
    return ctx.declared_log_y if card.log_y is None else bool(card.log_y)


def _log_y_unusable(card: Card, ctx: Context) -> str:
    return "" if _log_y_shown(card, ctx) else ctx.log_y_refusal


def _points_shown(card: Card, ctx: Context) -> bool:
    return ctx.declared_points if card.points is None else bool(card.points)


#: Every way of shaping a card, in the order a card shows them.
SHAPINGS: tuple[Shaping, ...] = (
    Shaping("x", "X", "list", 1, kinds=("chart",),
            options=lambda card, ctx: [(w, v, "") for w, v in ctx.axes.items()],
            shown=lambda card, ctx: card.x or "own",
            keywords=lambda card, ctx: {"x": card.x or "own"}),
    Shaping("x", "X", "list", 1, kinds=("numbers",),
            options=lambda card, ctx: [(w, v, "") for w, v in ctx.axes.items()],
            unusable=lambda card, ctx: "" if ctx.axes else "Nothing differs between the shown runs",
            shown=lambda card, ctx: card.x,
            keywords=lambda card, ctx: {"vs": card.x}),
    Shaping("log_x", "Log x", "switch", 1, unusable=_log_x_unusable,
            shown=lambda card, ctx: card.log_x,
            keywords=lambda card, ctx: {"log_x": card.log_x}),
    Shaping("log_y", "Log y", "switch", 1, unusable=_log_y_unusable, shown=_log_y_shown,
            keywords=lambda card, ctx: {"log_y": _log_y_shown(card, ctx)}),
    Shaping("colour_by", "Colour by", "one", 2, kinds=("chart",),
            options=lambda card, ctx: [("Run", "run", ""), ("Quantity", "quantity", "")],
            unusable=_colour_unusable, shown=lambda card, ctx: card.colour_by,
            keywords=lambda card, ctx: ({"colour_by": card.colour_by}
                                        if not _colour_unusable(card, ctx) else {})),
    # Its colours travel even when greyed: none named is one colour over every run.
    Shaping("by", "Colour by", "several", 2, kinds=("numbers",),
            options=lambda card, ctx: [(w, v, "") for w, v in ctx.telling.items()],
            unusable=lambda card, ctx: "" if ctx.telling else "Nothing else tells the runs apart",
            shown=lambda card, ctx: (tuple(ctx.telling.values()) if card.by is None
                                     else tuple(b for b in card.by if b in ctx.telling.values())),
            keywords=lambda card, ctx: {"by": (tuple(ctx.telling.values()) if card.by is None
                                               else tuple(b for b in card.by
                                                          if b in ctx.telling.values()))}),
    Shaping("runs", "Runs", "one", 2, kinds=("chart",), options=_chart_runs_options,
            unusable=_one_run, shown=_chart_runs,
            keywords=lambda card, ctx: {"runs": _chart_runs(card, ctx)}),
    Shaping("runs", "Runs", "one", 2, kinds=("numbers",),
            options=lambda card, ctx: [("All", "each", ""), ("Mean", "mean", ""),
                                       ("RMS", "rms", "")],
            unusable=lambda card, ctx: "" if ctx.pools else "No colour holds two runs at one value",
            shown=_numbers_runs,
            keywords=lambda card, ctx: {"runs": _numbers_runs(card, ctx)}),
    Shaping("points", "", "one", 3,
            options=lambda card, ctx: [("Dots", True, ""), ("Lines", False, "")],
            shown=_points_shown,
            keywords=lambda card, ctx: {"points": _points_shown(card, ctx)}),
    Shaping("table", "", "one", 4,
            options=lambda card, ctx: [("Chart", False, ""), ("Table", True, "")],
            shown=lambda card, ctx: card.table,
            keywords=lambda card, ctx: {}),
)


def offered(card: Card, ctx: Context) -> list[tuple[Shaping, str]]:
    """``(shaping, why it is greyed or "")`` for every way the card shapes what it draws, in
    order -- every one, on every card of its kind."""
    return [(s, s.unusable(card, ctx)) for s in SHAPINGS if ctx.kind in s.kinds]


def draw_keywords(card: Card, ctx: Context) -> dict[str, Any]:
    """What the card draws, as the figure layer's keywords (``figure.chart.chart_of`` for a
    chart card, ``sweep_of`` for a numbers card): its chart is drawn from them, and its still."""
    out: dict[str, Any] = {}
    for s, _ in offered(card, ctx):
        out.update(s.keywords(card, ctx))
    return out
