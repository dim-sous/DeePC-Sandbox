"""The Lab's page on Panel -- one :class:`Lab` per browser session, one screen.

The piece's runs on the left (a list to tick, filter and delete from, and the ticked runs
compared column by column), their cards on the right -- each the
figure layer's own drawing of what the records declare: a panel over the shown runs, or a summary
number against the setting that varies -- a draggable split between the two, the form over the
Config's groups in a drawer that opens from "New run" or a run's "use", and the terminal's own
output in a drawer along the bottom. Rows come from :mod:`lab.view`, a card's controls from
:mod:`lab.shaping`, what a control offers from :mod:`lab.form`, the commands from
:mod:`lab.jobs`, everything listed from :mod:`lab.state`, every chart from :mod:`figure.chart`
and :func:`figure.twin.figure_of`, the look from :mod:`lab.skin`. Served on 127.0.0.1 by
:func:`serve`.

Everything on the page is derived from what is on disk, what is ticked and what each card's
controls hold, on every event, in :meth:`Lab.render`; the one memo is the loaded records, keyed
by their file's mtime. The page computes nothing and restates no rule: a greyed control is its
field's own gate or the figure layer's refusal, a refusal is the terminal's words, a chart is the
figure layer's. A click that raises, or a page that cannot be built, shows its error on the page.
"""

from __future__ import annotations

import dataclasses
import functools
import html
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import panel as pn

from figure.chart import chart_of, distinct, pool_refusal, sweep_of
from figure.style import series_color
from figure.twin import figure_of
from record import capital

from . import jobs, shaping, skin, state, view
from .compare import CompareTable
from .form import Form, as_text
from .runlist import RunList
from .split import DEFAULT_PX, Split

_POLL_MS = 700
_CONSOLE_START = "Nothing has been run in this session."
_PLOT_CONFIG = {"displaylogo": False, "responsive": True, "displayModeBar": "hover",
                "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toImage"]}
_SANS = skin.SANS.replace('"', "'")
_MONO = f"font-family:{skin.MONO.replace(chr(34), chr(39))}"
_DIM = f"color:{skin.INK3}"
_RED = f"color:{skin.SIGNAL}"
_H2 = (f"font:600 12px {_SANS};letter-spacing:.05em;text-transform:uppercase;"
       f"color:{skin.INK2}")
_LABEL = f"{_DIM};font-size:12px;line-height:24px;display:inline-block"
#: A card's controls' margin (RCWS): with Panel's own (5, 10), 20 px between two buttons, and a
#: number card's controls over 160 runs took two lines of a 1050 px card.
_TOOL_MARGIN = (2, 3)
#: The width of the column a chart group's words stand in, before its charts: the form's label
#: track (RCWS: in 100 px the longest group name took three lines).
_GROUP_PX = skin.LABEL_PX
#: The columns the chart list's index of groups is read down (RCWS).
_INDEX_COLUMNS = 3
#: The words of the compare table's settings choice, and what each asks for (``view.SETTINGS``).
_SETTINGS = {"None": "none", "Differing": "differ", "All": "all"}


def _shown_in_page(exc: BaseException) -> None:
    """A callback that raises says so on the page and in the terminal. Not a fallback."""
    traceback.print_exc()
    if pn.state.notifications is not None:
        pn.state.notifications.error(f"{type(exc).__name__}: {exc}", duration=0)


def _toast(words: str) -> None:
    if pn.state.notifications is not None:
        pn.state.notifications.info(words, duration=3200)


def _every(callback: Callable[[], None], period_ms: int) -> Any:
    return pn.state.add_periodic_callback(callback, period=period_ms)


def _html(text: str, *, title: str = "", style: str = "", **kw: Any) -> pn.pane.HTML:
    t = f' title="{html.escape(title, quote=True)}"' if title else ""
    s = f' style="{style}"' if style else ""
    kw.setdefault("margin", (2, 5))      # a grid cell passes 0: there the gaps are the grid's
    return pn.pane.HTML(f"<span{t}{s}>{text}</span>", **kw)


def _h2(words: str) -> pn.pane.HTML:
    return _html(html.escape(words), style=_H2)


def _widget(cls: Any, **kw: Any) -> Any:
    """A Panel widget wearing the Lab's stylesheet."""
    return cls(stylesheets=[skin.WIDGET], **kw)


def _tool(cls: Any, **kw: Any) -> Any:
    """A control on a card: the Lab's stylesheet, compact."""
    kw.setdefault("margin", _TOOL_MARGIN)
    return cls(stylesheets=[skin.WIDGET, skin.TOOL], **kw)


class _Rows:
    """A list setting declaring ``rows`` drawn as labelled rows: one box per item, labelled by
    its index (``x1' =``), ✕ to take it out, a button to add one. It holds the list as the form
    holds every setting, as the text ``--set`` takes (JSON); ``""`` is the declared default.
    Every edit by hand calls ``on_edit`` with that text; a value written to it does not. Greying
    and the changed mark reach the drawn boxes in place: redrawing them while one is typed in
    would take the box away from under the keyboard."""

    def __init__(self, f: dict[str, Any], on_edit: Callable[[str], None]) -> None:
        self.f, self._on_edit = f, on_edit
        self.view = pn.Column(sizing_mode="stretch_width", margin=0)
        self._disabled, self._stylesheets = False, [skin.WIDGET]
        self._boxes: list[Any] = []
        self._buttons: list[Any] = []
        self._items: list[str] = [as_text(v) for v in f["default"] or []]
        self._draw()

    @property
    def value(self) -> str:
        return self._text()

    @value.setter
    def value(self, text: str) -> None:
        self._items = [as_text(v) for v in (json.loads(text) if text else self.f["default"] or [])]
        self._draw()

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, off: bool) -> None:
        self._disabled = off
        for widget in self._boxes + self._buttons:
            widget.disabled = off

    @property
    def stylesheets(self) -> list[Any]:
        return self._stylesheets

    @stylesheets.setter
    def stylesheets(self, sheets: list[Any]) -> None:
        self._stylesheets = list(sheets)
        for box in self._boxes:
            box.stylesheets = self._stylesheets

    def _text(self) -> str:
        """The rows as JSON. A list of numbers (θ) writes each row that reads as a number as
        one, so its default reads as unset, and any other row as the text typed, which the
        terminal refuses in its own words: never a row split or dropped on the way."""
        if self.f.get("texts"):
            return json.dumps(self._items, ensure_ascii=False)
        def number(text: str) -> Any:
            try:
                return float(text)
            except ValueError:
                return text
        return json.dumps([number(v) for v in self._items], ensure_ascii=False)

    def _draw(self) -> None:
        label, add = self.f["rows"]["label"], self.f["rows"]["add"]
        rows, self._boxes, self._buttons = [], [], []
        for i, item in enumerate(self._items):
            box = pn.widgets.TextInput(value=item, margin=0, sizing_mode="stretch_width",
                                       disabled=self._disabled, stylesheets=self._stylesheets)
            box.param.watch(lambda e, i=i: self._typed(i, e.new), "value_input")
            drop = _widget(pn.widgets.Button, label="✕", width=30, margin=(0, 0, 0, 6),
                           disabled=self._disabled)
            drop.on_click(lambda _e, i=i: self._drop(i))
            self._boxes.append(box)
            self._buttons.append(drop)
            rows.append(pn.Row(_html(html.escape(label.format(i + 1) + " ="), margin=0,
                                     style=f"{_MONO};font-size:12.5px;display:block;width:64px"),
                               box, drop, sizing_mode="stretch_width", margin=(0, 0, 4, 0)))
        more = _widget(pn.widgets.Button, label=add, width=120, margin=(2, 0, 0, 64),
                       disabled=self._disabled)
        more.on_click(lambda _e: self._add())
        self._buttons.append(more)
        self.view.objects = rows + [more]

    def _typed(self, i: int, text: str) -> None:
        self._items[i] = text
        self._on_edit(self._text())

    def _drop(self, i: int) -> None:
        del self._items[i]
        self._draw()
        self._on_edit(self._text())

    def _add(self) -> None:
        self._items.append("")
        self._draw()
        self._on_edit(self._text())


def _width(words: str) -> int:
    """A button as wide as its words, at the card's 12 px (RCWS)."""
    return 20 + 13 * len(words) // 2


def _tab(name: str, value: bool) -> pn.widgets.Toggle:
    """A chart's toggle, across the cell that holds it, in the controls' look (RCWS)."""
    return pn.widgets.Toggle(label=name, value=value, margin=0, sizing_mode="stretch_width",
                             css_classes=["tab"], stylesheets=[skin.WIDGET, skin.TOOL])


def _tab_cell(toggle: pn.widgets.Toggle, note: str) -> pn.Column:
    """A chart's toggle with its note under it: at least as wide as its name in the type the
    toggles are set in, and growing with the others on its line until the line is full (RCWS:
    cells as wide as the longest name left most of each line empty)."""
    said = ([_html(html.escape(note), style=f"{_DIM};font-size:11px", margin=(1, 0, 0, 6))]
            if note else [])
    return pn.Column(toggle, *said, margin=(3, 3), width=22 + 8 * len(toggle.label),
                     sizing_mode="fixed", styles={"flex": "1 1 auto", "max-width": "none"})


def _colour(i: int) -> str:
    """The colour the i-th shown run's lines wear on one plot (``figure.style.series_color``)."""
    return series_color(i)


class Lab:
    """The page's state for one browser session. ``launch``, ``every`` and ``toast`` are
    injectable so the page can be driven in Python (the tests do)."""

    def __init__(self, root: str, out_dir: str, *,
                 launch: Callable[..., jobs.Job] = jobs.launch,
                 every: Callable[[Callable[[], None], int], Any] = _every,
                 toast: Callable[[str], None] = _toast) -> None:
        self.root, self.out_dir = root, out_dir
        self.launch, self.every, self.toast = launch, every, toast
        self.pieces = state.pieces()
        self.piece = self.pieces[0]
        self.form = Form(self.piece)
        self.records: list[dict[str, Any]] = []
        self.job: Optional[jobs.Job] = None
        self._ticker: Any = None
        self._controls: dict[str, Any] = {}
        self._labels: dict[str, pn.pane.HTML] = {}
        self._tips: dict[str, pn.widgets.TooltipIcon] = {}
        self._group_notes: dict[str, pn.pane.HTML] = {}
        #: the folded groups opened, by name; kept across redraws of one piece's form
        self._unfolded: set[str] = set()
        self._filling = False
        self._from: Optional[str] = None            # the run the form was last loaded from
        self.shown: list[str] = []                  # the runs the cards and the table show
        self.cards: list[shaping.Card] = []
        self._seq = 0
        self._asked_open: list[str] = []            # what the address asked to open
        self.settings_shown = "none"
        #: path -> (mtime, record): a loaded record is read once per version of its file
        #: per file: the version read, the record (``None`` when refused, with the words why) and
        #: what it draws -- its charts and their groups, read once with it (``_drawable``)
        self._loaded: dict[str, tuple[float, Any, str, Optional[tuple]]] = {}
        #: card id -> the runs its chart draws, in the order shown
        self.drawn: dict[int, list[str]] = {}
        self._answer: Optional[Callable[[], None]] = None   # what the dialog's yes does
        self._history: list[str] = []               # the terminal's finished lines, as HTML
        self._folded: Optional[jobs.Job] = None     # the job whose output is in the history
        self._read_url()

        # -- header
        self.piece_select = _widget(pn.widgets.Select, width=170,
                                    options={p["label"]: p["name"] for p in self.pieces},
                                    value=self.piece["name"])
        self.piece_select.param.watch(lambda e: self._on_piece(e.new), "value")
        self.job_tag = _html("Idle", style=_DIM)
        self.stop_btn = _widget(pn.widgets.Button, label="Stop", width=58, disabled=True)
        self.stop_btn.on_click(lambda _e: self._stop())
        self.where = _html("")
        self.new_run_btn = _widget(pn.widgets.Button, label="New run", color="primary", width=92)
        self.new_run_btn.on_click(lambda _e: self.open_drawer())
        self.refresh = _widget(pn.widgets.Button, label="Refresh", width=72)
        self.refresh.on_click(lambda _e: self.reload())

        # -- the left pane: the runs, the shown ones compared
        self.runs_count = _html("", style=_DIM, stylesheets=[skin.BAR_FILL])
        self.run_filter = _tool(pn.widgets.TextInput, placeholder="Filter runs",
                                sizing_mode="stretch_width")
        self.run_filter.param.watch(lambda _e: self._draw_list(), "value_input")
        self.all_btn = _tool(pn.widgets.Button, label="All", width=_width("All"),
                             description="Tick every run the filter lists")
        self.all_btn.on_click(lambda _e: self.tick_listed())
        self.none_btn = _tool(pn.widgets.Button, label="None", width=_width("None"),
                              description="Untick every run the filter lists")
        self.none_btn.on_click(lambda _e: self.untick_listed())
        self.run_list = RunList(rows=[], shown=[], sizing_mode="stretch_width", styles=skin.RUNS)
        self.run_list.on_use = self.open_drawer
        self.run_list.on_delete = self._ask_delete_run
        self.run_list.param.watch(lambda e: self._on_shown(list(e.new)), "shown")
        self.compare_count = _html("", style=_DIM, stylesheets=[skin.BAR_FILL])
        self.settings_choice = _tool(pn.widgets.RadioButtonGroup, options=_SETTINGS,
                                     value="none")
        self.settings_choice.param.watch(lambda e: self._on_settings(e.new), "value")
        self.compare_table = CompareTable(sizing_mode="stretch_width",
                                          empty="No run shown: tick one above",
                                          corner="Column by column",
                                          styles={"background": skin.PAPER,
                                                  "border-radius": "6px", "overflow": "auto"})
        self.compare_table.on_number = self.open_number

        # -- the right pane: the cards
        self.open_count = _html("", style=_DIM)
        self.tabs = pn.Column(sizing_mode="stretch_width", margin=0)
        self._chart_toggles: dict[str, pn.widgets.Toggle] = {}
        self._index_toggles: dict[str, pn.widgets.Toggle] = {}
        self._lines: dict[str, Any] = {}
        #: what the list was last built from (:meth:`_draw_tabs`); the groups the index shows,
        #: and the cards whose group was shown when they opened
        self._tabs_hold: Optional[tuple] = None
        self._groups_shown: set[str] = set()
        self._groups_met: set[int] = set()
        self.show_all = _tool(pn.widgets.Button, label="Show all", width=_width("Show all"),
                              description="Show every group's charts")
        self.show_all.on_click(lambda _e: self._show_groups(True))
        self.hide_all = _tool(pn.widgets.Button, label="Hide all", width=_width("Hide all"),
                              description="Hide every group's charts; their names stay in the "
                                          "index")
        self.hide_all.on_click(lambda _e: self._show_groups(False))
        self.cards_box = pn.Column(sizing_mode="stretch_width")

        # -- deleting asks first, in a dialog of its own: nothing else can be clicked meanwhile
        self.question = pn.pane.HTML("")
        self.yes = _widget(pn.widgets.Button, label="Delete", color="danger", width=72)
        self.yes.on_click(lambda _e: self._on_yes())
        self.no = _widget(pn.widgets.Button, label="Cancel", width=72)
        self.no.on_click(lambda _e: self._on_no())
        self.asking = pn.Modal(self.question, pn.Row(self.no, self.yes), show_close_button=False)

        # -- the drawer: the form
        self.form_box = pn.Column(sizing_mode="stretch_width")
        self.from_words = _html("", style=_DIM)
        self.form_note = _html("")
        self.sweep_field = _widget(pn.widgets.Select, width=170)
        self.sweep_values = _widget(pn.widgets.TextInput, width=150, placeholder="v1, v2, …")
        self.sweep_field.param.watch(lambda e: self._on_sweep(field=e.new), "value")
        # A text box is read on `value_input` alone, every keystroke: a browser message that
        # carries both of a box's values at once (a paste, a key and Tab together) reached one
        # watcher over both as two events, and the box showed a value the run did not use
        # (2026-10-27, headless Chromium; RCWS watches `value_input` alone).
        self.sweep_values.param.watch(lambda e: self._on_sweep(values=e.new), "value_input")
        self.run_name = _widget(pn.widgets.TextInput, width=200)
        self.run_name.param.watch(lambda e: self._on_name(e.new), "value_input")
        #: On a line of its own under the bar, not in it: in the bar a long overwrite list pushed
        #: Run off the drawer, and made to give way there it was cut to two letters (2026-10-06).
        self.name_note = _html("", style=_DIM)
        self.run_btn = _widget(pn.widgets.Button, label="Run", color="primary", width=70)
        self.run_btn.on_click(lambda _e: self._run())
        self.clear_btn = _widget(pn.widgets.Button, label="Clear", width=62)
        self.clear_btn.on_click(lambda _e: self._clear())
        self.close_btn = _widget(pn.widgets.Button, label="Close", width=64)
        self.close_btn.on_click(lambda _e: self.close_drawer())
        self.command_line = pn.pane.HTML("", sizing_mode="stretch_width")
        self.banner = pn.pane.HTML("", visible=False, styles={"color": skin.SIGNAL})
        self.drawer = pn.Column(
            pn.Row(_html("<b style='font-size:15px'>New run</b>"), self.from_words,
                   pn.layout.HSpacer(), self.clear_btn, self.close_btn,
                   styles=skin.DRAWER_HEAD, stylesheets=[skin.BAR],
                   sizing_mode="stretch_width"),
            pn.Column(self.form_box, styles=skin.DRAWER_BODY, sizing_mode="stretch_both",
                      scroll=True),
            pn.Column(pn.Row(_html("Name", style=_DIM), self.run_name,
                             _html("Sweep", style=_DIM), self.sweep_field, self.sweep_values,
                             pn.layout.HSpacer(), self.run_btn, stylesheets=[skin.BAR],
                             sizing_mode="stretch_width"),
                      self.name_note, self.command_line, self.banner, self.form_note,
                      styles=skin.DRAWER_FOOT, sizing_mode="stretch_width"),
            visible=False, styles=skin.DRAWER, width=skin.DRAWER_PX)

        # -- the terminal
        self.terminal_btn = _widget(pn.widgets.Button, label="▸ Terminal", width=96,
                                    css_classes=["quiet"])
        self.terminal_btn.on_click(lambda _e: self._terminal(not self.console.visible))
        self.last_line = _html("Idle", style=f"{_MONO};font-size:12px;{_DIM}")
        self.console_text = pn.pane.HTML(f"<pre>{_CONSOLE_START}</pre>",
                                         stylesheets=[skin.CONSOLE], sizing_mode="stretch_width")
        self.console = pn.Column(self.console_text, scroll=True, height=260, visible=False,
                                 styles=skin.CONSOLE_BOX, sizing_mode="stretch_width")
        self.terminal = pn.Column(
            pn.Row(self.terminal_btn, self.last_line, styles=skin.TERMINAL_BAR,
                   stylesheets=[skin.BAR],
                   sizing_mode="stretch_width"),
            self.console, styles=skin.TERMINAL, sizing_mode="stretch_width")

        self._draw_form()
        self.reload()

    # ------------------------------------------------------------------ reading
    def reload(self) -> None:
        """Re-read the records on disk and redraw everything that shows them."""
        self.records = state.records(self.root, self.out_dir)
        self._offer_saved()
        # the piece's own: with two pieces, every piece's count stood beside one piece's folder
        n = len(self.runs())
        self.where.object = (f"<span style='{_MONO};font-size:12px;{_DIM}'>"
                             f"{html.escape(self.out_dir)}/{html.escape(self.piece['name'])}"
                             f" · {n} record{'s' if n != 1 else ''}</span>")
        readable = [r for r in self.runs() if not r["stale"]]
        names = {r["name"] for r in readable}
        kept = [name for name in self.shown if name in names]
        if not kept and readable:
            # the newest that can be read: a row the list greys has no tick to take it off with
            kept = [readable[0]["name"]]
        self.shown = kept
        declared = [row["name"] for row in view.chart_rows(self._shown_drawable(), [])]
        for name in self._asked_open:
            if name in declared and name not in self.open_charts:
                self._open(name)
        self._asked_open = []
        if not self.cards and declared:
            self._open(declared[0])                 # so the page is never empty
        self._draw_names()
        self.render()

    def runs(self) -> list[dict[str, Any]]:
        return [r for r in self.records if r["piece"] == self.piece["name"]]

    def shown_runs(self) -> list[dict[str, Any]]:
        """The ticked runs, in the order they were ticked."""
        by_name = {r["name"]: r for r in self.runs()}
        return [by_name[n] for n in self.shown if n in by_name]

    @property
    def open_charts(self) -> list[str]:
        """The panels open as cards, in the order they were opened."""
        return [c.chart for c in self.cards if c.kind == "chart"]

    def _record_of(self, run: dict[str, Any]) -> tuple[Any, str]:
        """The record behind a row, read once per version of its file -- a refusal too, else a
        run the class refuses was read again at every redraw (code review, 2026-10-27); else
        ``(None, why)`` in the loading's own words."""
        hit = self._loaded.get(run["path"])
        if hit is not None and hit[0] == run["mtime"]:
            return hit[1], hit[2]
        try:
            record, why = state.load(run), ""
        except (ValueError, OSError) as exc:
            record, why = None, str(exc)
        drawn = None
        if record is not None:
            series = record.series()
            drawn = ([s.name for s in series], {s.name: s.group for s in series},
                     {s.name: s.title for s in series})
        self._loaded[run["path"]] = (run["mtime"], record, why, drawn)
        return record, why

    def render(self) -> None:
        """Everything that shows the runs, from disk, the ticks and the cards, in one pass."""
        self._draw_list()
        self._draw_compare()
        self._draw_tabs()
        self._draw_cards()
        self._draw_url()

    # ------------------------------------------------------------------ the runs
    def _listed(self) -> list[dict[str, Any]]:
        return view.matching(self.runs(), self.run_filter.value_input or "")

    def _draw_list(self) -> None:
        listed = self._listed()
        colour = {name: _colour(i) for i, name in enumerate(self.shown)}
        self.run_list.rows = [{"name": r["name"], "stale": r["stale"],
                               "colour": colour.get(r["name"], "")}
                              for r in view.run_rows(listed)]
        self.run_list.empty = ("No run holds every word typed" if self.runs()
                               else f"No records yet: {self.piece['produced_by']}")
        if list(self.run_list.shown) != self.shown:
            self.run_list.shown = list(self.shown)
        n = len(self.runs())
        self.runs_count.object = (f"<span style='{_DIM}'>{n} on disk · {len(self.shown)} shown"
                                  + (f" · {len(listed)} listed" if len(listed) != n else "")
                                  + "</span>")

    def _on_shown(self, names: list[str]) -> None:
        if names == self.shown:
            return
        readable = {r["name"] for r in self.runs() if not r["stale"]}
        self.shown = [n for n in names if n in readable]
        self.render()

    def tick_listed(self) -> None:
        """Tick every run the filter lists that can be read, in the order listed."""
        for r in self._listed():
            if not r["stale"] and r["name"] not in self.shown:
                self.shown.append(r["name"])
        self.render()

    def untick_listed(self) -> None:
        """Untick every run the filter lists; a run it does not list keeps its tick."""
        listed = {r["name"] for r in self._listed()}
        self.shown = [n for n in self.shown if n not in listed]
        self.render()

    def _on_settings(self, value: str) -> None:
        self.settings_shown = value
        self._draw_compare()

    def _draw_compare(self) -> None:
        runs = self.shown_runs()
        constants = []
        for run in runs:
            record, _why = self._record_of(run)
            constants.append(getattr(record, "constants", None) if record is not None else None)
        rows = view.compare_rows(runs, self.form.fields, settings=self.settings_shown,
                                 constants=constants, declared=self._numbers_declared())
        self.compare_count.object = (f"<span style='{_DIM}'>"
                                     f"{html.escape(view.compare_count(rows, runs, self.form.fields))}"
                                     f"</span>")
        opened = {c.number for c in self.cards if c.kind == "numbers"}
        self.compare_table.param.update(
            heads=[r["name"] for r in runs],
            rows=[{**row, "open": row["kind"] == "number" and row["key"] in opened}
                  for row in rows])

    # ------------------------------------------------------------------ deleting
    def _ask_delete_run(self, name: str) -> None:
        self._ask(f"Delete {self.out_dir}/{self.piece['name']}/{name}.json?",
                  functools.partial(self._delete, records=[name]))

    def _ask(self, question: str, then: Callable[[], None]) -> None:
        self.question.object = f"<p style='font-size:14px'>{html.escape(question)}</p>"
        self._answer = then
        self.asking.show()

    def _on_yes(self) -> None:
        """Do what was asked, once: a second click that reaches here after the first has
        answered finds nothing left to do."""
        then, self._answer = self._answer, None
        self.asking.hide()
        if then is not None:
            then()

    def _on_no(self) -> None:
        self._answer = None
        self.asking.hide()

    def _delete(self, records: tuple = ()) -> None:
        try:
            gone = jobs.delete(self.piece["name"], records, root=self.root, out_dir=self.out_dir)
        except ValueError as exc:
            self._print([f"refused: {exc}"], kind="refused")
            self.toast(f"Not deleted: {exc}")
            return
        print("lab: deleted " + " ".join(gone))
        self._print([f"lab: deleted {path}" for path in gone], kind="cmd")
        self.shown = [n for n in self.shown if n not in records]
        self.toast(f"Deleted {' '.join(gone)}")
        self.reload()

    # ------------------------------------------------------------------ the chart list
    def _numbers_declared(self) -> list[dict[str, Any]] | None:
        """The numbers the first shown run's record declares, read off it through its class --
        so a file written before its numbers were grouped is compared in its class's groups;
        ``None`` (the files' own headers) when it does not load."""
        runs = self.shown_runs()
        record = self._record_of(runs[0])[0] if runs else None
        if record is None:
            return None
        return [dataclasses.asdict(n) for n in record.number_specs()]

    def _drawable(self, run: dict[str, Any]) -> dict[str, Any]:
        """``run`` with the charts it can draw and their groups as its record's class declares
        them now, read through the class (:meth:`_record_of`, once per version of its file). Read
        off the file, a chart change left every saved run's list behind: the list offered charts
        the code no longer drew and none it now did (RCWS #257). A run that does not load keeps
        its file's list; its card says why."""
        record = self._record_of(run)[0]
        if record is None:
            return run
        hit = self._loaded.get(run["path"])
        if hit is not None and hit[1] is record:
            panels, groups, titles = hit[3]          # read once with the record
        else:
            series = record.series()
            panels, groups = [s.name for s in series], {s.name: s.group for s in series}
            titles = {s.name: s.title for s in series}
        return {**run, "panels": panels, "panel_groups": groups, "panel_titles": titles}

    def _shown_drawable(self) -> list[dict[str, Any]]:
        return [self._drawable(run) for run in self.shown_runs()]

    def _draw_tabs(self) -> None:
        """The charts the shown runs declare, each a toggle, in the groups and the order their
        records' class declares them: an index of the groups by name, then each group on lines
        of its own, its words before it, while its name in the index shows it. This list
        re-sorts nothing.

        The toggles are built again only when what the list holds changes -- its charts, their
        groups and notes, whether a run is shown; anything else sets the toggles already there.
        Built again on every click, a click landing while the page drew the new ones was lost:
        2 of 5 charts clicked 80 ms apart in headless Chromium (2026-10-27; RCWS #256, 4 of 5)."""
        rows = view.chart_rows(self._shown_drawable(), self.open_charts)
        holds = tuple((r["name"], r["title"], r["group"], r["sub"]) for r in rows)
        if holds != self._tabs_hold:
            self._tabs_hold = holds
            self._build_tabs(rows)
        self._set_tabs(rows)

    def _build_tabs(self, rows: list[dict[str, Any]]) -> None:
        """The list's toggles, the index and each group's line, built and put on the page. Each
        toggle as wide as its name with its note under it, growing with the others on its line
        until the line is full; each group on lines of its own, its words in a column before it
        (RCWS ``_build_tabs``, the developer's: groups flowing on from one another mixed on a
        line)."""
        self._chart_toggles, self._lines, self._index_toggles = {}, {}, {}
        cells: dict[str, list[Any]] = {}
        for group, in_group in view.grouped(rows):
            for row in in_group:
                toggle = _tab(row["title"], row["open"])
                toggle.param.watch(lambda e, n=row["name"]: self._on_chart(n, e.new), "value")
                self._chart_toggles[row["name"]] = toggle
                cells.setdefault(group, []).append(_tab_cell(toggle, row["sub"]))
        named = [group for group in cells if group]
        self.show_all.visible = self.hide_all.visible = bool(named)
        if not rows:
            items: list[Any] = [_html("Tick a run to see what it can draw", style=_DIM)]
        elif not named:
            items = [pn.FlexBox(*cells.get("", []), align_items="flex-start", margin=0,
                                gap="0 2px", sizing_mode="stretch_width")]
        else:
            for group, in_line in cells.items():
                self._lines[group] = pn.FlexBox(
                    _html(html.escape(group), style=_LABEL, width=_GROUP_PX, margin=(3, 4, 0, 2)),
                    pn.FlexBox(*in_line, align_items="flex-start", margin=0, gap="0 2px",
                               styles={"flex": "1 1 0", "min-width": "0"}),
                    flex_wrap="nowrap", align_items="flex-start", margin=0,
                    sizing_mode="stretch_width")
            items = [self._index(named), *self._lines.values()]
        self.tabs.objects = items

    def _set_tabs(self, rows: list[dict[str, Any]]) -> None:
        """What the list's toggles say: which charts are open, which groups the index shows, and
        how many charts are open of how many. A toggle set to what its card already is changes
        nothing (:meth:`_on_chart`, :meth:`_on_group` return)."""
        for row in rows:
            self._chart_toggles[row["name"]].value = row["open"]
        if self._index_toggles:
            shown = self._groups_on_show(rows)
            for group, toggle in self._index_toggles.items():
                on = group in shown
                toggle.param.update(value=on, label=f"{'▾' if on else '▸'} {group}")
            for group, line in self._lines.items():
                line.visible = not group or group in shown
        self.open_count.object = (f"<span style='{_DIM}'>{len(self.open_charts)} of "
                                  f"{len(rows)} open</span>")

    def _groups_on_show(self, rows: list[dict[str, Any]]) -> set[str]:
        """The groups the list shows below its index: the ones the developer showed, and the
        group of every chart opened since the list was last drawn -- by its toggle, by the
        address, or as the first chart a page opens on. Hiding a group closes nothing."""
        group_of = {row["name"]: row["group"] for row in rows}
        for card in self.cards:
            if card.kind == "chart" and card.id not in self._groups_met and card.chart in group_of:
                self._groups_met.add(card.id)
                self._groups_shown.add(group_of[card.chart])
        return self._groups_shown

    def _index(self, named: list[str]) -> pn.FlexBox:
        """The list's groups by name, read down :data:`_INDEX_COLUMNS` columns, each name a
        toggle that shows its group's lines below -- "▾" while it does, "▸" while not
        (:meth:`_set_tabs`). RCWS #255: 45 charts in 15 groups, every group shown, put the first
        chart below the bottom of a 1700 x 1100 window."""
        per_column = -(-len(named) // _INDEX_COLUMNS)

        def entry(group: str) -> pn.widgets.Toggle:
            toggle = pn.widgets.Toggle(label=f"▸ {group}", value=False, margin=0,
                                       sizing_mode="stretch_width", css_classes=["index"],
                                       stylesheets=[skin.WIDGET, skin.INDEX])
            toggle.param.watch(lambda e, g=group: self._on_group(g, e.new), "value")
            self._index_toggles[group] = toggle
            return toggle

        columns = [pn.Column(*(entry(g) for g in named[c * per_column:(c + 1) * per_column]),
                             margin=0, styles={"flex": "1 1 0", "min-width": "0"})
                   for c in range(_INDEX_COLUMNS)]
        return pn.FlexBox(*columns, flex_wrap="nowrap", align_items="flex-start", gap="0 16px",
                          margin=(0, 2, 6, 2), sizing_mode="stretch_width", styles=skin.INDEX_BOX)

    def _on_group(self, group: str, show: bool) -> None:
        if show == (group in self._groups_shown):
            return
        if show:
            self._groups_shown.add(group)
        else:
            self._groups_shown.discard(group)
        self._draw_tabs()

    def _show_groups(self, every: bool) -> None:
        """show all / hide all: every group the list holds on show below its index, or none."""
        self._groups_shown.clear()
        if every:
            self._groups_shown.update(group for group in self._lines if group)
        self._draw_tabs()

    def group_toggles(self) -> dict[str, pn.widgets.Toggle]:
        """The index's toggles on screen, by group."""
        return dict(self._index_toggles)

    def toggles(self) -> dict[str, pn.widgets.Toggle]:
        """The chart toggles on screen, by panel -- a hidden group's among them."""
        return dict(self._chart_toggles)

    def _open(self, chart: str) -> shaping.Card:
        self._seq += 1
        card = shaping.Card(self._seq, "chart", chart=chart)
        self.cards.append(card)
        return card

    def _on_chart(self, panel: str, open_it: bool) -> None:
        opened = next((c for c in self.cards if c.kind == "chart" and c.chart == panel), None)
        if open_it and opened is None:
            self._open(panel)
        elif not open_it and opened is not None:
            self.cards.remove(opened)
        else:
            return
        self._draw_tabs()
        self._draw_cards()
        self._draw_url()

    def open_number(self, number: str) -> None:
        """A number of the compare table, as a card of its own over the shown runs; asked
        again, the card closes."""
        opened = next((c for c in self.cards if c.kind == "numbers" and c.number == number), None)
        if opened is not None:
            self.cards.remove(opened)
        else:
            self._seq += 1
            self.cards.insert(0, shaping.Card(self._seq, "numbers", number=number))
        self._draw_compare()
        self._draw_cards()

    # ------------------------------------------------------------------ the cards
    def _pairs(self, names: Optional[list[str]] = None) -> tuple[list[tuple[str, Any]], list[str]]:
        """``(name, record)`` for every shown run that loads, and, for every one that does not,
        why -- in the loader's own words."""
        pairs, words = [], []
        for run in self.shown_runs():
            record, why = self._record_of(run)
            if record is None:
                words.append(f"{run['name']}: {why}")
            else:
                pairs.append((run["name"], record))
        return pairs, words

    def _draw_cards(self) -> None:
        """One card per open chart and number: the numbers first, then the charts in the order
        the records declare them."""
        runs = self.shown_runs()
        self.drawn = {}
        if not runs:
            self.cards_box.objects = [_html("Tick a run to see its charts", style=_DIM)]
            return
        if not self.cards:
            self.cards_box.objects = [_html("Open a chart above, or click a number in the "
                                            "compare table", style=_DIM)]
            return
        declared = [row["name"] for row in view.chart_rows([self._drawable(r) for r in runs],
                                                            self.open_charts)]
        place = {name: i for i, name in enumerate(declared)}
        numbers = [c for c in self.cards if c.kind == "numbers"]
        charts = sorted((c for c in self.cards if c.kind == "chart"),
                        key=lambda c: (place.get(c.chart, len(place)), c.id))
        self.cards_box.objects = [self._card_view(card) for card in numbers + charts]

    def _card_view(self, card: shaping.Card) -> pn.Column:
        pairs, refused = self._pairs()
        if card.kind == "numbers":
            return self._numbers_view(card, pairs, refused)
        return self._chart_view(card, pairs, refused)

    def _frame(self, card: shaping.Card, title: str, doc: str, tools: list[Any], body: Any,
               foot: str, warn: bool) -> pn.Column:
        """A card: its title (its description on the title's hover) and close at the line's
        end; then its controls, one row of groups, a whole group wrapping onto the next line when
        the card is too narrow for it; then what it draws; then a foot -- why it could not draw.
        The description is not repeated under the title (RCWS ``_frame``, the developer's: it
        said what the hover said)."""
        close = _widget(pn.widgets.Button, label="✕", width=34, description="Close this card")
        close.on_click(lambda _e: self._close(card))
        # the panel's name rides on the title as data-panel: the tests found a card by its
        # title's text, which found none once a title showed words, not the name (2026-10-27)
        top = pn.FlexBox(_html(f"<b data-panel='{html.escape(card.chart or card.number)}' "
                               f"style='{_MONO};font-size:15px'>{html.escape(title)}</b>",
                               title=doc, stylesheets=[skin.BAR_FILL]),
                         close,
                         flex_wrap="nowrap", align_items="center", margin=0,
                         stylesheets=[skin.BAR], sizing_mode="stretch_width")
        items: list[Any] = [top]
        if tools:
            items.append(pn.FlexBox(*tools, flex_wrap="wrap", align_items="center", margin=0,
                                    gap="4px 8px", stylesheets=[skin.BAR],
                                    sizing_mode="stretch_width"))
        items.append(body)
        if foot:
            items.append(_html(html.escape(foot), title=foot,
                               style=f"font-size:12.5px;{_RED if warn else _DIM}"))
        return pn.Column(*items, styles=skin.CARD, sizing_mode="stretch_width")

    def _toolbar(self, card: shaping.Card, ctx: shaping.Context) -> list[Any]:
        """The card's controls, one per way it shapes what it draws, in the groups and the order
        :data:`lab.shaping.SHAPINGS` declares, each group one box that stays whole -- every
        control on every card, greyed with the reason on its hover where it cannot be used
        right now. A greyed control is its words and nothing to click, the reason its hover:
        a disabled Panel widget would show none."""
        groups: dict[int, list[Any]] = {}
        for s, why in shaping.offered(card, ctx):
            value, options = s.shown(card, ctx), s.options(card, ctx)
            cell: list[Any] = []
            if s.words:
                cell.append(_html(html.escape(s.words), style=_LABEL, margin=(0, 2, 0, 4)))
            if why:
                said = ([next((w for w, v, _ in options if v == value), "")] if s.kind == "list"
                        else [" | ".join(w for w, _, _ in options)])
                cell += [_html(html.escape(w), title=capital(why), style=skin.GREYED,
                               margin=_TOOL_MARGIN)
                         for w in said if w]
            elif s.kind == "list":
                shows = next((w for w, v, _ in options if v == value), "")
                select = _tool(pn.widgets.Select, options={w: v for w, v, _ in options},
                               value=value if value in [v for _, v, _ in options] else None,
                               width=max(80, min(200, 40 + 8 * len(shows))))
                select.param.watch(functools.partial(self._on_shaping, card, s.field), "value")
                cell.append(select)
            elif s.kind == "one":
                for words, option, greyed in options:
                    if greyed:
                        cell.append(_html(html.escape(words), title=capital(greyed),
                                          style=skin.GREYED,
                                          margin=_TOOL_MARGIN))
                        continue
                    toggle = _tool(pn.widgets.Toggle, label=words, value=option == value,
                                   width=_width(words))
                    toggle.param.watch(functools.partial(self._on_one, card, s.field, option),
                                       "value")
                    cell.append(toggle)
            else:
                raise ValueError(f"a card control of kind {s.kind!r}: lab.shaping declares "
                                 f"'list' and 'one'")
            groups.setdefault(s.group, []).extend(cell)
        # Each group as wide as its controls and kept whole: a FlexBox stretches across what
        # holds it by default, which put each group on a line of its own (seen 2026-10-27 at
        # 1440 px; RCWS `_cell`).
        return [pn.FlexBox(*groups[g], flex_wrap="nowrap", align_items="center", margin=0,
                           sizing_mode=None, styles=skin.GROUP if i else {},
                           stylesheets=[skin.BAR])
                for i, g in enumerate(sorted(groups))]

    def _on_shaping(self, card: shaping.Card, name: str, event: Any) -> None:
        setattr(card, name, event.new)
        self._draw_cards()

    def _on_one(self, card: shaping.Card, name: str, option: Any, event: Any) -> None:
        """One of a few: pressed, the card takes it; pressed off, nothing changes (one of them
        is always held)."""
        if event.new:
            setattr(card, name, option)
        self._draw_cards()

    def _chart_context(self, card: shaping.Card, pairs: list[tuple[str, Any]]) -> shaping.Context:
        """What the chart card's controls are read off: the figure layer's own answers."""
        held = [(n, r) for n, r in pairs if card.chart in [s.name for s in r.series()]]
        if not held:
            return shaping.Context("chart", runs=0)
        pool = {how: pool_refusal(held, card.chart, how) for how in ("mean", "rms")}
        return shaping.Context("chart", runs=len(held), pool=pool)

    def _chart_view(self, card: shaping.Card, pairs: list[tuple[str, Any]],
                    refused: list[str]) -> pn.Column:
        """A panel over the shown runs, drawn by the figure layer from what the card's controls
        hold; a line for each run that does not load, in the loader's words. A shown run that
        declares no such panel is the figure layer's refusal, in its words: drawn without it, the
        comparison would leave it out unsaid."""
        ctx = self._chart_context(card, pairs)
        declared = next((s for _, r in pairs for s in r.series() if s.name == card.chart), None)
        body: Any = pn.pane.HTML("")
        foot, warn = " · ".join(refused), bool(refused)
        if ctx.runs:
            try:
                panel = chart_of(pairs, card.chart, **shaping.draw_keywords(card, ctx))
            except ValueError as exc:
                foot, warn = capital(str(exc)), True
            else:
                body = self._body(card, panel)
                self.drawn[card.id] = [n for n, _ in pairs]
        return self._frame(card, declared.title if declared else card.chart,
                           declared.doc if declared else "", self._toolbar(card, ctx), body,
                           foot, warn)

    def _numbers_context(self, card: shaping.Card,
                         pairs: list[tuple[str, Any]]) -> shaping.Context:
        fields = {f["name"]: f for f in self.form.fields}
        vary = distinct(pairs) if len(pairs) > 1 else []
        if card.x not in vary:
            card.x = vary[0] if vary else ""
        axes = {view.term(fields[v]): v for v in vary}
        ctx = shaping.Context("numbers", runs=len(pairs), axes=axes)
        if not card.x:
            return ctx
        try:
            plain = sweep_of(pairs, card.number, vs=card.x)
        except ValueError:
            return ctx
        pools = any(len(set(line.x)) < len(line.x) for line in plain.axes[0].lines)
        return shaping.Context(**{**ctx.__dict__, "pools": pools})

    def _numbers_view(self, card: shaping.Card, pairs: list[tuple[str, Any]],
                      refused: list[str]) -> pn.Column:
        """A summary number against the setting that varies, every shown run a point, drawn by
        the figure layer from what the card's controls hold."""
        ctx = self._numbers_context(card, pairs)
        spec = next((n for _, r in pairs for n in r.number_specs() if n.name == card.number), None)
        title = spec.label if spec else card.number
        fields = {f["name"]: f for f in self.form.fields}
        if card.x:
            title = f"{title} vs {fields[card.x]['label']}"
        body: Any = pn.pane.HTML("")
        foot, warn = " · ".join(refused), bool(refused)
        if not card.x:
            foot, warn = ("One run shown, or nothing differs between the shown runs: tick runs "
                          "that differ in a setting"), True
        else:
            try:
                panel = sweep_of(pairs, card.number, **shaping.draw_keywords(card, ctx))
            except ValueError as exc:
                foot, warn = capital(str(exc)), True
            else:
                body = self._body(card, panel)
                self.drawn[card.id] = [n for n, _ in pairs]
        return self._frame(card, title, spec.doc if spec else "", self._toolbar(card, ctx), body,
                           foot, warn)

    def _body(self, card: shaping.Card, panel: Any) -> Any:
        """What a card shows: the figure layer's chart, as its live twin."""
        # a number card draws no legend: the hover names each point's run (RCWS)
        fig = figure_of(panel, legend=card.kind != "numbers")
        return pn.pane.Plotly(fig, config=_PLOT_CONFIG, sizing_mode="stretch_width",
                              height=fig["layout"]["height"])

    def _close(self, card: shaping.Card) -> None:
        if card in self.cards:
            self.cards.remove(card)
        self._draw_tabs()
        self._draw_cards()
        self._draw_compare()
        self._draw_url()

    # ------------------------------------------------------------------ the url
    def _read_url(self) -> None:
        """``?shown=a,b&open=soc`` on the way in: a hand-off is a link."""
        location = pn.state.location
        if location is None:
            return
        query = location.query_params or {}
        self.shown = [n for n in str(query.get("shown", "")).split(",") if n]
        self._asked_open = [p for p in str(query.get("open", "")).split(",") if p]

    def _draw_url(self) -> None:
        location = pn.state.location
        if location is None:
            return
        try:
            location.update_query(shown=",".join(self.shown), open=",".join(self.open_charts))
        except Exception as exc:                          # the address bar is a convenience
            print(f"lab: the address could not be updated: {exc!r}", file=sys.stderr)

    # ------------------------------------------------------------------ the form
    def open_drawer(self, from_run: Optional[str] = None) -> None:
        """Open the form; from a run's ``use``, loaded with that run's settings."""
        if from_run is not None:
            self._use(from_run)
        self.from_words.object = (f"<span style='{_DIM}'>From "
                                  f"{html.escape(self._from) if self._from else 'defaults'}"
                                  f"</span>")
        self.drawer.visible = True

    def close_drawer(self) -> None:
        self.drawer.visible = False

    def _draw_form(self) -> None:
        self._controls.clear(); self._labels.clear(); self._tips.clear()
        self._group_notes.clear()
        groups: dict[str, list[Any]] = {}
        blocks: dict[str, list[Any]] = {}       # the row fields, full width under the grid
        for f in self.form.fields:
            cells = groups.setdefault(f["group"], [])
            if f.get("rows"):
                blocks.setdefault(f["group"], []).append(self._rows_block(f))
            else:
                cells.extend(self._field_cells(f))
        sections = []
        for g, cells in groups.items():
            note = _html("", style=_DIM)
            self._group_notes[g] = note
            grid = ([pn.GridBox(*cells, ncols=6, stylesheets=[skin.FORM_GRID],
                                sizing_mode="stretch_width")] if cells else [])
            body = pn.Column(*grid, *blocks.get(g, []), sizing_mode="stretch_width", margin=0)
            head: Any = _h2(g)
            if all(f.get("folded") for f in self.form.fields if f["group"] == g):
                body.visible = g in self._unfolded
                head = pn.widgets.Toggle(label=f"{'▾' if body.visible else '▸'} {g}",
                                         value=body.visible, margin=0, css_classes=["index"],
                                         stylesheets=[skin.WIDGET, skin.INDEX])
                head.param.watch(lambda e, g=g, body=body, head=head:
                                 self._on_fold(g, body, head, e.new), "value")
            sections.append(pn.Column(pn.Row(head, note, stylesheets=[skin.BAR],
                                             sizing_mode="stretch_width"),
                                      body, sizing_mode="stretch_width", margin=(6, 0)))
        self.form_box.objects = sections
        self.sweep_field.options = {"none": "", **{words: n for n, words in self.form.axes()}}
        self._apply_gates()

    def _on_fold(self, group: str, body: pn.Column, head: Any, show: bool) -> None:
        """A folded group opened or closed: its settings shown or not, the arrow saying which."""
        (self._unfolded.add if show else self._unfolded.discard)(group)
        body.visible = show
        head.label = f"{'▾' if show else '▸'} {group}"

    def _offer_saved(self) -> None:
        """Each piece's readable saved runs, as the paths a setting naming one offers, from the
        records just read; each such pick refilled, holding what it held."""
        saved: dict[str, list[str]] = {}
        for r in self.records:
            if not r["stale"]:
                saved.setdefault(r["piece"], []).append(
                    f"{self.out_dir}/{r['piece']}/{r['name']}.json")
        self.form.saved = saved
        for f in self.form.fields:
            control = self._controls.get(f["name"])
            if not f.get("record_of") or control is None:
                continue
            self._filling = True
            try:
                control.options = {o["words"]: o["value"] for o in self.form.options(f) or []}
                control.value = self.form.set.get(f["name"], "")
            finally:
                self._filling = False
            if f["name"] in self._tips:
                self._tips[f["name"]].value = self._line_of(f, control.value)

    def _rows_block(self, f: dict[str, Any]) -> Any:
        """A row field: its label (the help on its hover), then its rows (:class:`_Rows`)."""
        name = f["name"]
        label = _html("", margin=0)
        self._labels[name] = label
        editor = _Rows(f, on_edit=lambda text, n=name: self._on_type(n, text))
        editor.value = self.form.set.get(name, "")
        self._controls[name] = editor
        return pn.Column(label, editor.view, sizing_mode="stretch_width", margin=(0, 0, 8, 0))

    def _field_cells(self, f: dict[str, Any]) -> list[Any]:
        """A field as the THREE cells of a six-track grid: label, box, unit (or the pick's
        tooltip, or nothing). The cells go into the grid themselves rather than into a row of
        their own, because a row is its own flex box and two rows share no geometry -- which is
        what left the boxes 196px apart across the sections, measured 2026-10-05. Every cell is
        margin-free: the spacing is the grid's gaps, so the tracks are what they say they are."""
        name = f["name"]
        label = _html("", margin=0)
        self._labels[name] = label
        options = self.form.options(f)
        after: Any = pn.Spacer(height=0)             # the third track is never left unfilled
        if f.get("unit"):
            after = _html(html.escape(f["unit"]), margin=0,
                          style=f"{_MONO};font-size:11px;{_DIM}")
        if options is None:
            control = _widget(pn.widgets.TextInput, margin=0, sizing_mode="stretch_width",
                              placeholder=self.form.shown_default(f),
                              value=self.form.set.get(name, ""))
            control.param.watch(lambda e, n=name: self._on_type(n, e.new), "value_input")
        else:
            control = _widget(pn.widgets.Select, margin=0, sizing_mode="stretch_width",
                              options={o["words"]: o["value"] for o in options},
                              value=self.form.set.get(name, ""))
            control.param.watch(lambda e, n=name: self._on_pick(n, e.new), "value")
            after = pn.widgets.TooltipIcon(value=self._line_of(f, control.value), margin=0)
            self._tips[name] = after
        self._controls[name] = control
        return [label, control, after]

    def _line_of(self, f: dict[str, Any], value: str) -> str:
        for o in self.form.options(f) or []:
            if o["value"] == value:
                return o["line"] or o["words"]
        return ""

    def _apply_gates(self) -> None:
        """Every control's state is its field's own gate or the sweep row's axis, nothing else."""
        changed: dict[str, int] = {}
        for f in self.form.fields:
            name = f["name"]
            off, words = self.form.state_of(f)
            control, label = self._controls[name], self._labels[name]
            control.disabled = off
            control.stylesheets = [skin.WIDGET, skin.CHANGED] if name in self.form.set \
                else [skin.WIDGET]
            style = (f"display:block;text-align:left;font-size:12.5px;"
                     f"color:{skin.INK if name in self.form.set else skin.INK2};"
                     + ("opacity:.45" if off else ""))
            label.object = (f"<span title=\"{html.escape(words, quote=True)}\" style=\"{style}\">"
                            f"{html.escape(f['label'])}</span>")      # the unit sits by the box
            if name in self._tips:
                self._tips[name].value = self._line_of(f, control.value)
            changed[f["group"]] = changed.get(f["group"], 0) + (name in self.form.set)
        for g, note in self._group_notes.items():
            total = sum(1 for f in self.form.fields if f["group"] == g)
            moved = changed.get(g, 0)
            note.object = (f"<span style='color:{skin.ACCENT if moved else skin.INK3};"
                           f"font-size:12px'>"
                           + (f"{moved} of {total} changed" if moved
                              else f"{total} at defaults") + "</span>")
        note_words, warns = self.form.note()
        self.form_note.object = (f"<span style='{_RED if warns else _DIM}'>"
                                 f"{html.escape(note_words)}</span>")
        self._draw_typed()

    def _fill_controls(self) -> None:
        """Push the form's values into the widgets without the watchers reading it as typing; a
        saved run's pick is offered again first, so the path it now holds is one it offers."""
        self._offer_saved()
        self._filling = True
        try:
            for name, control in self._controls.items():
                self._write(control, self.form.set.get(name, ""))
            self.sweep_field.value = self.form.sweep_field
            self._write(self.sweep_values, self.form.sweep_values)
            self._write(self.run_name, self.form.name)
        finally:
            self._filling = False
        self._apply_gates()
        self._draw_names()

    @staticmethod
    def _write(control: Any, text: str) -> None:
        """Put ``text`` in a control and in every parameter a watcher reads, so what the box
        shows and what the run would use cannot drift apart."""
        control.value = text
        if hasattr(control, "param") and "value_input" in control.param:
            control.value_input = text

    def _draw_names(self) -> None:
        self.run_name.placeholder = self.form.free_name(self.runs())
        note = self.form.name_note(self.runs())
        self.name_note.object = f"<span style='{_RED if note else _DIM}'>{html.escape(note)}</span>"
        self._draw_typed()

    def _draw_typed(self) -> None:
        """The line Run would type, live, or why it would be refused -- the terminal's words."""
        overrides, sweeps = self.form.launch_words()
        name = self.form.name.strip() or self.form.free_name(self.runs())
        try:
            line = jobs.line_of(jobs.run_argv(self.piece["name"], overrides, name, self.out_dir,
                                              sweeps))
        except ValueError as exc:
            self.command_line.object = (f"<span style='{_MONO};font-size:12px;{_RED}'>"
                                        f"{html.escape(str(exc))}</span>")
            return
        self.command_line.object = (f"<span style='{_MONO};font-size:12px;color:{skin.INK2}'>"
                                    f"lab: <b style='color:{skin.INK}'>{html.escape(line)}</b>"
                                    f"</span>")

    # ------------------------------------------------------------------ typing
    def _on_type(self, name: str, text: str) -> None:
        if self._filling:
            return
        taken = self.form.type_or_pick(name, text)
        self._taken_back(taken)
        self._apply_gates()

    def _on_pick(self, name: str, value: str) -> None:
        if self._filling:
            return
        taken = self.form.type_or_pick(name, value)
        self._taken_back(taken)
        self._apply_gates()

    def _on_sweep(self, *, field: Optional[str] = None, values: Optional[str] = None) -> None:
        if self._filling:
            return
        taken = self.form.sweep(self.form.sweep_field if field is None else field,
                                self.form.sweep_values if values is None else values)
        self._taken_back(taken)
        self._apply_gates()
        self._draw_names()                      # a sweep renames what the run would write

    def _taken_back(self, names: list[str]) -> None:
        """A box behind a gate that just shut is shown empty, not holding a value Run would refuse."""
        if not names:
            return
        self._filling = True
        try:
            for n in names:
                self._write(self._controls[n], "")
        finally:
            self._filling = False

    def _on_name(self, text: str) -> None:
        if self._filling:
            return
        self.form.name = text
        self._draw_names()

    def _use(self, run_name: str) -> None:
        run = next((r for r in self.runs() if r["name"] == run_name), None)
        if run is None:
            self._say(f"no record {run_name!r} any more; press refresh")
            return
        n, dropped = self.form.use(run["config"])
        self._from = run_name
        self._fill_controls()
        words = f"{n} setting{'s' if n != 1 else ''} loaded from {run_name}"
        if dropped:
            words += f"; {', '.join(dropped)} no longer declared"
        if self.form.taken:
            words += f"; {len(self.form.taken)} behind a closed gate taken back"
        self.toast(words)

    def _clear(self) -> None:
        self.form.clear()
        self._from = None
        self.from_words.object = f"<span style='{_DIM}'>From defaults</span>"
        self._fill_controls()

    def _on_piece(self, name: str) -> None:
        self.piece = next(p for p in self.pieces if p["name"] == name)
        self.form = Form(self.piece)
        # every one of these is keyed by a bare run name, which belongs to the piece it came from
        self.shown, self.cards, self._loaded, self._from = [], [], {}, None
        self._groups_shown, self._groups_met, self._tabs_hold = set(), set(), None
        self._unfolded = set()
        self._draw_form()
        self.reload()

    # ------------------------------------------------------------------ running
    def _say(self, words: str) -> None:
        self.banner.object = f"<b>{html.escape(words)}</b>"
        self.banner.visible = bool(words)

    def _run(self) -> None:
        if self.job is not None and self.job.running:
            self.toast("A run is in progress — stop it first")
            return
        overrides, sweeps = self.form.launch_words()
        name = self.form.name.strip() or self.form.free_name(self.runs())
        try:
            argv = jobs.run_argv(self.piece["name"], overrides, name, self.out_dir, sweeps)
        except ValueError as exc:
            self._say(f"refused: {exc}")
            return
        line = jobs.line_of(argv)
        try:
            job = self.launch(argv, cwd=self.root)
        except ValueError as exc:                  # one `deepc run` per process
            self._say(f"refused: {exc}")
            return
        self._say("")
        print(f"lab: {line}")
        self._fold()
        self._print([f"lab: {line}"], kind="cmd")
        self.job = job
        self.close_drawer()
        self._terminal(True)
        self._draw_job()
        if self._ticker is None:
            self._ticker = self.every(self._poll, _POLL_MS)

    def _stop(self) -> None:
        if self.stop_job():
            self.toast("Stopping…")

    def stop_job(self) -> bool:
        """Terminate this session's run, if one is going. The stop button calls it, and so does
        the end of the browser session: a closed tab otherwise leaves a full run burning CPU."""
        return self.job is not None and self.job.stop()

    def _poll(self) -> None:
        """Every 700 ms while a job runs: redraw its output; when it ends, list what it wrote.

        Its own failures are shown here: Panel's periodic callback has no exception hook, so an
        error raised inside it would otherwise reach only the server's log.
        """
        try:
            self._tick()
        except Exception as exc:
            _shown_in_page(exc)
            self._say(f"reading what the run wrote failed: {exc}")

    def _tick(self) -> None:
        job = self.job
        self._draw_job()
        if job is None or job.running:
            return
        if self._ticker is not None:
            stop = getattr(self._ticker, "stop", None)
            if callable(stop):
                stop()
            self._ticker = None
        wrote = state.wrote(job.output, self.root, self.out_dir)
        for path in wrote:                          # show what just ran
            name = Path(path).stem
            if name not in self.shown:
                self.shown.append(name)
        self.last_line.object = (f"<span style='{_MONO};font-size:12px;{_DIM}'>"
                                 f"{html.escape(job.line)} · {html.escape(self._outcome(job, len(wrote)))}"
                                 f"</span>")
        self.reload()
        self.toast(f"{self.piece['label']}: {self._outcome(job, len(wrote))}")

    @staticmethod
    def _outcome(job: jobs.Job, wrote: int) -> str:
        """What the finished job did, counting what it WROTE as well as how it ended: a sweep
        can write some of its records and then be refused, and a job the page stopped refused
        nothing."""
        made = f"wrote {wrote} record{'s' if wrote != 1 else ''}"
        if job.stopped:
            return f"stopped after it {made}"
        if job.returncode == 0:
            return made
        if wrote:
            return f"{made}, then ended with exit {job.returncode}"
        return "the run was refused"

    def _draw_job(self) -> None:
        job = self.job
        if job is None:
            return
        self._draw_console()
        if job.running:
            self.job_tag.object = (f"<span style='color:{skin.LIVE}'>●</span> <span style='{_DIM}'>"
                                   f"Running…</span> <span style='{_MONO};font-size:12px'>"
                                   f"{html.escape(job.line)}</span>")
        else:
            tag = "Done" if job.returncode == 0 else f"Ended with exit {job.returncode}"
            self.job_tag.object = (f"<span style='{_DIM if job.returncode == 0 else _RED}'>"
                                   f"{tag}</span>")
        self.run_btn.disabled = job.running
        self.stop_btn.disabled = not job.running

    # ------------------------------------------------------------------ the terminal
    def _terminal(self, open_it: bool) -> None:
        self.console.visible = open_it
        self.terminal_btn.label = ("▾" if open_it else "▸") + " Terminal"

    def _print(self, lines: list[str], *, kind: str = "") -> None:
        """Lines into the terminal, as they are; ``kind`` is how they are coloured."""
        for line in lines:
            words = html.escape(line.rstrip("\n"))
            self._history.append(f"<span class='{kind}'>{words}</span>" if kind else words)
        self._draw_console()

    def _fold(self) -> None:
        """The finished job's output joins the history, once, before the next command."""
        job = self.job
        if job is not None and job is not self._folded and not job.running:
            self._history.extend(html.escape(line.rstrip("\n")) for line in job.output)
            self._folded = job

    def _draw_console(self) -> None:
        live = []
        job = self.job
        if job is not None and job is not self._folded:
            live = [html.escape(line.rstrip("\n")) for line in job.output]
        text = "\n".join(self._history + live)
        self.console_text.object = f"<pre>{text or _CONSOLE_START}</pre>"
        self.console.scroll_position = 10 ** 7 + len(text)

    # ------------------------------------------------------------------ the page
    def page(self) -> pn.Column:
        header = pn.Row(
            _html("<b style='font-size:15px;letter-spacing:.01em'>DeePC Lab</b>"),
            self.piece_select, self.job_tag, self.stop_btn,
            pn.layout.HSpacer(), self.where, self.refresh, self.new_run_btn,
            styles=skin.TOP, stylesheets=[skin.BAR], sizing_mode="stretch_width")
        left = pn.Column(
            pn.Row(_h2("Runs"), self.runs_count, stylesheets=[skin.BAR],
                   sizing_mode="stretch_width"),
            pn.Row(self.run_filter, self.all_btn, self.none_btn, stylesheets=[skin.BAR],
                   sizing_mode="stretch_width"),
            self.run_list,
            pn.Row(_h2("Compare"), self.compare_count, stylesheets=[skin.BAR],
                   sizing_mode="stretch_width", margin=(14, 0, 0, 0)),
            pn.Row(_html("Settings", style=_LABEL), self.settings_choice,
                   stylesheets=[skin.BAR], sizing_mode="stretch_width"),
            self.compare_table,
            styles=skin.LEFT, scroll=True, sizing_mode="stretch_width")
        right = pn.Column(
            pn.Row(_h2("Figures"), self.open_count, self.show_all, self.hide_all,
                   pn.layout.HSpacer(),
                   stylesheets=[skin.BAR], sizing_mode="stretch_width"),
            self.tabs,
            self.cards_box,
            styles=skin.PANE, scroll=True, sizing_mode="stretch_width")
        self.split = Split(left=left, right=right, left_px=DEFAULT_PX, styles=skin.BODY,
                           sizing_mode="stretch_width")
        # both ways: the drawer's body stretches both ways, and Panel widened the page to match
        # with a warning on every load (RCWS); declared, it is what the page already was
        return pn.Column(header, self.split, self.drawer, self.terminal, self.asking,
                         sizing_mode="stretch_both", margin=0)


def page(root: str, out_dir: str) -> pn.viewable.Viewable:
    """The page for one session, or -- when it cannot be built -- the traceback, on the page."""
    try:
        lab = Lab(root, out_dir)
        shown = lab.page()
        try:                                         # a closed tab must not leave a run going
            pn.state.on_session_destroyed(lambda _ctx: lab.stop_job())
        except Exception:                            # no session: driven in Python, or a test
            pass
        return shown
    except Exception:
        text = traceback.format_exc()
        print(text, file=sys.stderr)
        return pn.pane.HTML(f"<h3>the Lab could not build its page</h3><pre>{html.escape(text)}</pre>")


def serve(port: int, root: str, out_dir: str, *, show: bool = True, threaded: bool = False) -> Any:
    """Serve on 127.0.0.1 until ctrl-c (or return the server when ``threaded``)."""
    pn.extension("plotly", notifications=True, exception_handler=_shown_in_page)
    for css in (skin.FONTS, skin.PAGE):
        if css not in pn.config.raw_css:
            pn.config.raw_css.append(css)
    # Import every registered piece before the address answers rather than inside the first
    # page load: a piece can be slow to import and the Lab needs all of their field
    # declarations. Each session reads them again, out of `sys.modules`.
    state.pieces()
    return pn.serve({"/": lambda: page(root, out_dir)}, port=port, address="127.0.0.1",
                    websocket_origin=[f"127.0.0.1:{port}", f"localhost:{port}"], show=show,
                    title="DeePC Lab", threaded=threaded)
