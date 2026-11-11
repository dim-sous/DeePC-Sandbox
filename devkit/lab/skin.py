"""The Lab's look: tokens, fonts, and one stylesheet or style dict per kind of thing.

Nothing here computes. A *style dict* is what a layout box wears on its own element; a
*stylesheet* is what a widget, a pane or a component loads into its shadow root, where the
page's own CSS cannot reach (every Panel component renders into one). The paper, the inks and
the rules are the figure layer's own (:mod:`figure.style`): a chart sits on the same paper as
its card, unframed, and the two cannot drift apart. The look is RCWS's ``lab/skin.py``
(``feat/unequal-state-mixing``, 822b0cf), the developer's choice on 2026-10-27.
"""

from __future__ import annotations

from figure.style import FONT as SANS
from figure.style import GRID, INK, INK_SECONDARY, MONO, SURFACE

FONTS = ('@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
         '&family=IBM+Plex+Sans:wght@400;500;600&display=swap");')

GROUND, PAPER = "#f3f3f0", SURFACE
INK2, INK3 = INK_SECONDARY, "#8b8a84"
LINE, LINE2 = GRID, "#cfcec8"
ACCENT, ACCENT_INK, ACCENT_SOFT = "#0e6f8e", "#0a5a73", "#e2eef2"
SIGNAL, LIVE = "#a3342f", "#1f8a5a"

HEADER_PX = 48
TERMINAL_PX = 34

#: The document: fonts and the ground. Inherited properties (font, colour) reach into every
#: shadow root from here; non-inherited ones do not, which is what the stylesheets are for.
PAGE = f"""
html, body {{ height: 100%; }}
body {{ margin: 0; background: {GROUND}; color: {INK}; font: 14px/1.45 {SANS};
        -webkit-font-smoothing: antialiased; overflow: hidden; }}
"""

#: Inputs, selects, buttons, toggles and check boxes.
WIDGET = f"""
:host {{ font-family: {SANS}; }}
.bk-input {{ font-family: {MONO}; font-size: 12.5px; color: {INK}; background: {GROUND};
             border: 1px solid {LINE2}; border-radius: 4px; min-height: 26px; }}
.bk-input:focus {{ border-color: {ACCENT}; box-shadow: none; }}
.bk-input::placeholder {{ color: {INK3}; }}
.bk-input:disabled {{ opacity: .5; }}
.bk-btn {{ font-family: {SANS}; font-size: 12.5px; border-radius: 4px; padding: 4px 10px;
           line-height: 1.3; }}
.bk-btn-default {{ background: {PAPER}; color: {INK2}; border: 1px solid {LINE2}; }}
.bk-btn-default:hover {{ color: {INK}; border-color: {INK3}; }}
.bk-btn-default.bk-active {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {INK}; }}
.bk-btn-primary {{ background: {ACCENT}; border: 1px solid {ACCENT}; color: #fff;
                   font-weight: 500; padding: 5px 14px; }}
.bk-btn-primary:hover {{ background: {ACCENT_INK}; border-color: {ACCENT_INK}; }}
.bk-btn-danger {{ background: {SIGNAL}; border: 1px solid {SIGNAL}; color: #fff; }}
.bk-btn:disabled {{ opacity: .45; }}
:host(.tab) .bk-btn {{ font-family: {MONO}; padding: 0 10px; text-align: left;
                       justify-content: flex-start; }}
:host(.quiet) .bk-btn-default {{ border-color: transparent; background: transparent; }}
input[type=checkbox] {{ accent-color: {ACCENT}; }}
label {{ font-size: 12.5px; color: {INK2}; }}
"""

#: A control holding a value away from its default.
CHANGED = f".bk-input {{ border-left: 2px solid {ACCENT}; background: {PAPER}; }}"

#: The form's tracks: label, box, unit, twice across. Every section's grid is given these same
#: fixed widths, so a box in RUN sits under a box in ESTIMATOR -- tracks that size to content
#: line up only within one grid.
LABEL_PX, CONTROL_PX, UNIT_PX = 170, 150, 48
#: The form's grid. The cells carry no margin of their own (the gaps are the grid's), and
#: `align-items: center` is what puts a label that wrapped to two lines on the same centre as its
#: box. `!important` because the GridBox sets `grid-template-columns` itself from its `ncols`.
FORM_GRID = f"""
:host {{ grid-template-columns: repeat(2, {LABEL_PX}px {CONTROL_PX}px {UNIT_PX}px) !important;
         align-items: center; column-gap: 8px; row-gap: 4px; }}
:host > * {{ align-self: center !important; }}
"""

#: The words column's width, in per cent of the box the table scrolls in (``cqw`` -- of the box,
#: not of the table, so it holds when the table is wider than the box); the runs' columns share
#: the rest equally, each never narrower than ``RUN_MIN_PX`` -- a number of :func:`view.text`'s
#: longest form, ``-1.234e+05``, on one line.
TERMS_SHARE, RUN_MIN_PX = 40, 96
#: The compare table (the ``CompareTable`` component's shadow root), RCWS's: the columns are the
#: table's, never their contents' -- a fixed layout, every cell wrapping inside its column, and
#: the table no narrower than its runs' columns at ``RUN_MIN_PX``; past that the box scrolls
#: sideways under a words column that stays put. Laid out by its contents, the table was as wide
#: as its longest line: in RCWS a 52-character value widened its column past the pane's edge; in
#: BESS, four runs in the left pane's default width at a 1440 px window cut the third run's column
#: at the pane's edge (2026-10-27).
TABLE = f"""
.box {{ overflow-x: auto; container-type: inline-size; }}
table.cmp {{ border-collapse: collapse; table-layout: fixed; width: 100%; font-size: 12.5px;
             font-family: {SANS};
             min-width: calc({TERMS_SHARE}cqw + var(--runs) * {RUN_MIN_PX}px); }}
table.cmp col.terms {{ width: {TERMS_SHARE}cqw; }}
table.cmp th, table.cmp td {{ overflow-wrap: anywhere; }}
table.cmp th:first-child, table.cmp td:first-child {{ position: sticky; left: 0;
                                                     background: {PAPER}; }}
table.cmp th {{ background: {PAPER}; font-family: {MONO};
                font-weight: 500; font-size: 12px; text-align: right; padding: 8px 10px 6px;
                border-bottom: 1px solid {LINE}; color: {INK}; vertical-align: bottom; }}
table.cmp th:first-child {{ text-align: left; font-family: {SANS}; font-weight: 500;
                            font-size: 11px; letter-spacing: .04em; text-transform: uppercase;
                            color: {INK3}; }}
table.cmp td {{ padding: 4px 10px; border-bottom: 1px solid {LINE}; text-align: right;
                font-family: {MONO}; font-variant-numeric: tabular-nums; color: {INK2};
                line-height: 1.25; }}
table.cmp td:first-child {{ text-align: left; font-family: {SANS}; color: {INK2}; }}
table.cmp tr.differs td {{ color: {INK}; }}
table.cmp tr.head td {{ font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase;
                        color: {INK3}; padding-top: 12px; border-bottom: 0; font-family: {SANS}; }}
table.cmp tr.click td:first-child {{ cursor: pointer; }}
table.cmp tr.click:hover td:first-child {{ text-decoration: underline dotted;
                                           text-underline-offset: 3px; color: {INK}; }}
table.cmp tr.open td:first-child {{ font-weight: 600; color: {INK}; }}
table.cmp tr:last-child td {{ border-bottom: 0; }}
.empty {{ font-size: 13px; color: {INK3}; padding: 10px 12px; }}
"""

#: A group of a card's controls after the first: a rule before it -- the groups by what they
#: act on (RCWS, the developer's: lined up in two rows' columns, they were not grouped by sense).
GROUP = {"border-left": f"1px solid {LINE2}", "padding-left": "8px"}

#: A card's control it cannot use right now: its words, drawn as an off control greyed as a
#: disabled one is, the reason on its hover (RCWS ``lab/skin.py``: a disabled Panel widget
#: carries no hover of its own).
GREYED = (f"display:inline-block;box-sizing:border-box;height:24px;line-height:22px;"
          f"border:1px solid {LINE2};border-radius:4px;padding:0 9px;font-size:12px;"
          f"color:{INK2};opacity:.45;cursor:default;white-space:nowrap")

#: A card's controls and the chart toggles, over :data:`WIDGET` -- RCWS's, the developer's look:
#: one height for every control; off, an outline and grey words; on, filled in the accent with
#: white words; no shadow. Bokeh's own pressed button is sunken, a grey fill under an inset
#: shadow, and on and off read alike (seen on BESS's page, 2026-10-27). Forced: Panel's own rules
#: for a solid button group are more specific than these and drew a pressed one in grey (RCWS,
#: measured in the browser: rgb(243, 243, 243) under this sheet's accent).
TOOL = f"""
.bk-btn {{ height: 24px; padding: 0 9px; font-size: 12px; line-height: 22px;
           box-shadow: none !important; background-image: none !important; }}
.bk-btn-default {{ background: transparent !important; color: {INK2} !important;
                   border: 1px solid {LINE2} !important; }}
.bk-btn-default:hover {{ color: {INK} !important; border-color: {INK3} !important; }}
.bk-btn-default.bk-active, .bk-btn-default.bk-active:hover {{
    background: {ACCENT} !important; border-color: {ACCENT} !important; color: #fff !important; }}
.bk-btn:focus {{ outline: none; }}
.bk-input {{ height: 24px; min-height: 24px; padding: 0 6px; font-size: 12px; }}
"""

#: A name in the chart list's index of its groups (RCWS): the group's words as the list heads its
#: lines with them, not a chart's toggle -- dim while the group is hidden, inked while it shows,
#: wrapping where its column is narrower than the name.
INDEX = f"""
.bk-btn {{ height: auto; min-height: 20px; padding: 1px 2px; font-family: {SANS}; font-size: 12px;
           line-height: 18px; text-align: left; justify-content: flex-start;
           white-space: normal; color: {INK3} !important; background: transparent !important;
           border: none !important; box-shadow: none !important; }}
.bk-btn:hover {{ color: {INK} !important; }}
.bk-btn.bk-active, .bk-btn.bk-active:hover {{ color: {INK} !important;
                                             background: transparent !important; }}
.bk-btn:focus {{ outline: none; }}
"""
#: The index: a rule under it, between the names and the lines they show.
INDEX_BOX = {"border-bottom": f"1px solid {LINE}", "padding-bottom": "4px"}

#: The list on the left -- the runs (``lab.runlist``), RCWS's: a run's row is its name, a stale
#: run's reason under it.
LIST = f"""
.list {{ background: {PAPER}; border-radius: 6px; overflow: hidden; font-family: {SANS}; }}
.run {{ display: grid; grid-template-columns: 22px minmax(0, 1fr) auto auto; align-items: center;
        gap: 8px; padding: 8px 10px 8px 12px; border-bottom: 1px solid {LINE}; cursor: pointer; }}
.run:last-child {{ border-bottom: 0; }}
.run:hover {{ background: color-mix(in srgb, {ACCENT_SOFT} 45%, {PAPER}); }}
.dot {{ width: 12px; height: 12px; border-radius: 50%; border: 1.5px solid {LINE2};
        display: inline-block; box-sizing: border-box; }}
.top {{ display: flex; align-items: baseline; gap: 10px; min-width: 0; }}
.name {{ font-family: {MONO}; font-weight: 500; font-size: 13px; color: {INK2};
         white-space: nowrap; }}
.main .name {{ white-space: normal; overflow-wrap: anywhere; }}
.run.shown .name {{ color: {INK}; }}
.when {{ font-family: {MONO}; font-size: 11.5px; color: {INK3}; white-space: nowrap;
         overflow: hidden; text-overflow: ellipsis; }}
.why {{ font-size: 12px; color: {SIGNAL}; margin-top: 1px; }}
.run.stale {{ cursor: default; }}
.run.stale .name {{ color: {INK3}; }}
.act {{ visibility: hidden; font: 12px {SANS}; color: {INK2}; background: {PAPER};
        border: 1px solid {LINE2}; border-radius: 4px; padding: 3px 9px; cursor: pointer; }}
.run:hover .act, .act:focus-visible {{ visibility: visible; }}
.act:hover {{ color: {INK}; border-color: {INK3}; }}
.act.del:hover {{ color: {SIGNAL}; border-color: {SIGNAL}; }}
.empty {{ font-size: 13px; color: {INK3}; padding: 10px 12px; }}
"""

#: The split (the ``Split`` component's shadow root).
SPLIT = f"""
.split {{ display: grid; height: 100%; min-height: 0; }}
.handle {{ cursor: col-resize; position: relative; }}
.handle::after {{ content: ""; position: absolute; inset: 0 2px; border-radius: 2px; }}
.handle:hover::after, .handle.dragging::after {{ background: {ACCENT}; }}
"""

#: The terminal's output (a pre inside an HTML pane).
CONSOLE = f"""
pre {{ margin: 0; font: 12.5px/1.5 {MONO}; color: {INK2}; white-space: pre-wrap;
       word-break: break-word; }}
.cmd {{ color: {INK}; font-weight: 500; }}
.wrote {{ color: {ACCENT_INK}; }}
.refused {{ color: {SIGNAL}; }}
"""

# -- what the layout boxes wear
#: Every bar on the page: a row of text panes beside widgets, put on ONE centreline. A
#: STYLESHEET, not a style dict, and it sets `align-self` on the CHILDREN rather than
#: `align-items` on the row: Panel gives every component its own `align`, default `start`, and
#: emits it as `align-self` on the child, which beats any `align-items` the row carries. The row
#: is the child's shadow root, so only a stylesheet reaches them. Measured 2026-10-05: without
#: this a text pane (margin 2) sat 9-10px off the button beside it (margin 5).
BAR = """
:host { align-items: center; }
:host > * { align-self: center !important; }
"""
#: The stretchy text in a bar: it gives way before the controls beside it do, so a long one
#: shortens instead of pushing a control off the pane. A STYLESHEET: the words sit in an
#: inline-block the markup pane builds inside its own shadow root, and `text-overflow` neither
#: inherits into it nor elides an atomic inline, so the rule has to reach that box and make it
#: a block. `flex` is forced because the pane carries `flex: 0 0 auto` of its own.
BAR_FILL = """
:host { flex: 1 1 auto !important; min-width: 0; }
:host > div { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
"""
TOP = {"height": f"{HEADER_PX}px", "background": PAPER,
       "border-bottom": f"1px solid {LINE}", "padding": "0 20px", "box-sizing": "border-box"}
BODY = {"height": f"calc(100vh - {HEADER_PX}px)"}
#: A pane scrolls because its Column is declared ``scroll=True``, not by an overflow of its
#: own: Bokeh caps every child of a layout at ``max-height: 100%``, and Panel lifts the cap only
#: for a Column it knows scrolls. RCWS, scrolled by a style, measured at a 1258x775 window: a
#: 64-run list 3,436 px tall in an 894 px box, drawn over the compare table.
PANE = {"height": "100%", "box-sizing": "border-box",
        "padding": f"0 16px {TERMINAL_PX + 24}px"}
LEFT = {**PANE, "border-right": f"1px solid {LINE}"}
#: The runs list: at most this much of the window, scrolling inside, so the compare table stays
#: just under it however many runs there are (RCWS).
RUNS = {"max-height": "40vh", "overflow-y": "auto", "border-radius": "6px"}
CARD = {"background": PAPER, "border-radius": "6px", "padding": "10px 16px 8px",
        "margin-bottom": "12px"}
DRAWER_PX = 840           # two columns of label · box · unit, measured 2026-10-05
DRAWER = {"position": "fixed", "top": f"{HEADER_PX}px", "right": "0",
          "height": f"calc(100vh - {HEADER_PX}px)", "max-width": "100%", "background": PAPER,
          "border-left": f"1px solid {LINE}", "box-shadow": "-16px 0 40px rgba(0,0,0,.10)",
          "z-index": "6", "box-sizing": "border-box"}
DRAWER_HEAD = {"padding": "8px 20px", "border-bottom": f"1px solid {LINE}"}
#: The body takes what the head and foot leave, and scrolls; the drawer's own height comes from
#: its top and bottom being pinned.
DRAWER_BODY = {"padding": "4px 20px 16px", "overflow": "auto", "flex": "1 1 auto",
               "min-height": "0"}
DRAWER_FOOT = {"padding": "10px 20px 12px", "border-top": f"1px solid {LINE}", "background": PAPER}
TERMINAL = {"position": "fixed", "left": "0", "right": "0", "bottom": "0", "background": PAPER,
            "border-top": f"1px solid {LINE}", "z-index": "5"}
TERMINAL_BAR = {"height": f"{TERMINAL_PX}px", "padding": "0 20px", "box-sizing": "border-box"}
CONSOLE_BOX = {"border-top": f"1px solid {LINE}", "padding": "6px 20px 12px"}
