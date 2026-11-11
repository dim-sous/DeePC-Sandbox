"""The Lab driven in a real browser (headless Chromium), as a person drives it (BESS's
``tests/test_lab_browser.py``, adapted).

The Python-driven page tests (``test_lab_view.py``) set a widget's values one at a time; a
browser does not. On 2026-10-27 the suite passed (189) while the real page lost a value pasted
into a box or typed with Tab right after it: one browser message carried both of the box's
values, a watcher over both got two events from it, and the run kept its default while the box
showed the value. A view is validated by driving it (RCWS ``[A9]``), and only a browser drives
it as a person does. These tests hold, on the page as served:

* what is typed or pasted into the form is what the run would use;
* every chart's points in the browser are exactly its record's numbers;
* driving the page raises no error in the browser or on the server;
* at half a laptop screen and at full width, nothing runs off the page and no words on a chart
  sit on top of each other.

The browser is Playwright's Chromium: once per computer, ``uv run playwright install chromium``.
"""

from __future__ import annotations

import socket
import time
import urllib.request
from pathlib import Path

import numpy as np
import panel as pn
import pytest
from playwright import sync_api as playwright

from harness import get_component
from lab.app import serve
from tests.lab_runs import make_runs


#: Every chart in the page's shadow roots, as plotly holds it: each trace's name, x and y, and
#: the figure's meta (what panel it is).
_PLOTS = """() => {
  const out = [];
  const walk = root => root.querySelectorAll('*').forEach(el => {
    if (el.classList && el.classList.contains('js-plotly-plot') && el.data)
      out.push({meta: el.layout.meta, traces: el.data.map(t => ({name: t.name, x: t.x, y: t.y,
                                                                    fill: t.fill || ''}))});
    if (el.shadowRoot) walk(el.shadowRoot);
  });
  walk(document);
  return out;
}"""
#: The words the page shows, leaf by leaf, through its shadow roots.
_TEXT = """() => {
  const out = [];
  const walk = root => { root.querySelectorAll('*').forEach(el => { if (el.shadowRoot) walk(el.shadowRoot); });
    root.querySelectorAll('span,b,button,td,th,pre').forEach(e => {
      const t = (e.innerText || '').trim(); if (t && !e.children.length) out.push(t); }); };
  walk(document);
  return out;
}"""
#: On every chart, the boxes of the words plotly draws -- the legend, the annotations, the axis
#: titles -- as [chart, what, left, top, right, bottom].
_WORD_BOXES = """() => {
  const out = [];
  const walk = root => root.querySelectorAll('*').forEach(el => {
    if (el.classList && el.classList.contains('js-plotly-plot')) {
      const sel = ['.legend', '.annotation', '.g-xtitle', '[class^="g-y"][class$="title"]'];
      sel.forEach(s => el.querySelectorAll(s).forEach(w => { const r = w.getBoundingClientRect();
        if (r.width && r.height) out.push([el.layout.meta.panel, s, r.left, r.top, r.right, r.bottom]); }));
    }
    if (el.shadowRoot) walk(el.shadowRoot);
  });
  walk(document);
  return out;
}"""
#: Every element under the page's shadow roots whose right edge is past the window's.
_PAST_THE_EDGE = """() => {
  const out = [], w = window.innerWidth;
  const walk = root => root.querySelectorAll('*').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width && r.right > w + 1 && getComputedStyle(el).visibility !== 'hidden')
      out.push((el.className && el.className.baseVal === undefined ? el.className : el.tagName) + ' ' + Math.round(r.right));
    if (el.shadowRoot) walk(el.shadowRoot);
  });
  walk(document);
  return out;
}"""


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """The Lab served over two tiny real records, and a browser to drive it with."""
    root, out_dir = make_runs(tmp_path_factory.mktemp("lab-browser"))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    handler = pn.config.exception_handler
    server = serve(port, root, out_dir, show=False, threaded=True)
    deadline = time.time() + 30
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
            break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.2)
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:                      # no browser on this computer: say how
            pytest.fail(f"no Chromium for Playwright here ({str(exc).splitlines()[0]}); "
                        f"install it once: uv run playwright install chromium")
        try:
            yield f"http://127.0.0.1:{port}/", browser, root, out_dir
        finally:
            browser.close()
            server.stop()
            pn.config.exception_handler = handler      # `serve` sets it process-wide


def _page(browser, url: str, width: int = 1440):
    page = browser.new_page(viewport={"width": width, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(f"page: {e}"))
    page.on("console", lambda m: errors.append(f"console: {m.text}") if m.type == "error" else None)
    page.goto(url, wait_until="networkidle")
    page.wait_for_timeout(1500)
    return page, errors


def _text(page) -> list[str]:
    return page.evaluate(_TEXT)


def _settle(page, ms: int = 900) -> None:
    page.wait_for_timeout(ms)


def test_what_is_pasted_or_typed_with_tab_at_once_is_what_the_run_would_use(served) -> None:
    url, browser, *_ = served
    page, errors = _page(browser, url)
    try:
        page.locator("button", has_text="New run").click()
        _settle(page)
        # the boxes left to the code's own default, in the form's order: Duration, T_ini, N, Q,
        # R, λ_g, ...
        boxes = page.locator('input[placeholder="Code default"]')
        boxes.nth(0).click()                                     # the Duration box
        page.keyboard.insert_text("0.5")                         # a paste
        page.keyboard.press("Tab")                               # and Tab, at once
        _settle(page)
        boxes.nth(5).click()                                     # the λ_g box
        page.keyboard.type("3", delay=0)                         # a key and Tab, at once
        page.keyboard.press("Tab")
        _settle(page)
        line = next(t for t in _text(page) if t.startswith("uv run deepc run"))
        assert "--set sim_duration=0.5" in line and "--set lambda_g=3" in line, line
        assert not errors, errors
    finally:
        page.close()


def test_every_chart_in_the_browser_is_exactly_its_records_numbers(served) -> None:
    url, browser, root, out_dir = served
    page, errors = _page(browser, f"{url}?shown=second,first&open=output-e-x1,input-F1,cost")
    try:
        plots = page.evaluate(_PLOTS)
        assert sorted(p["meta"]["panel"] for p in plots) == ["cost", "input-F1", "output-e-x1"]
        records = {name: get_component("closed_loop").Record.load(
            Path(root) / out_dir / "closed_loop" / f"{name}.json") for name in ("first", "second")}
        checked = 0
        for plot in plots:
            for trace in plot["traces"]:
                # a long name is broken onto lines at its separators, each kept on its line
                name = trace["name"].replace("<br>", " ")
                run, _, label = name.partition(" — ")
                if trace["fill"] or run not in records:
                    continue                    # a band, a reference, a line every run holds
                series = next(s for s in records[run].series() if s.name == plot["meta"]["panel"])
                if not label:                   # a run left with one trace of its own: its name
                    label = next((t for t, v in series.traces
                                  if np.array_equal(np.asarray(v, float),
                                                    np.asarray(trace["y"], float),
                                                    equal_nan=True)), "")
                y = dict(series.traces).get(label)
                if y is None:
                    continue                                   # a reference held by one run
                assert np.array_equal(np.asarray(trace["x"], float), np.asarray(series.x, float))
                assert np.array_equal(np.asarray(trace["y"], float), np.asarray(y, float),
                                      equal_nan=True), \
                    f"{plot['meta']['panel']} {trace['name']}"
                checked += 1
        assert checked == 2 * (1 + 1 + 1), "every run's own line on the three charts"
        assert not errors, errors
    finally:
        page.close()


def test_driving_every_kind_of_card_raises_nothing_in_the_browser(served) -> None:
    url, browser, *_ = served
    page, errors = _page(browser, f"{url}?shown=second,first&open=input-F1")
    try:
        page.locator("td", has_text="Position combined RMSE").first.click()   # a number card
        _settle(page, 1500)
        # the closed loop's charts declare no statistic, so Mean and RMS are greyed, no button;
        # a chart card's one control is its runs (the developer's cut, 2026-11-11)
        for words in ("Split", "Overlay"):
            page.locator("button", has_text=words).first.click()
            _settle(page)
        for gone in ("Save", "Table", "Dots", "Log x", "Quantity"):
            assert not page.locator("button", has_text=gone).count(), f"{gone} is not on a card"
        page.locator("button", has_text="Differing").click()
        _settle(page)
        assert any("vs λ_g" in t for t in _text(page)), "the number card is on the page"
        assert not errors, errors
    finally:
        page.close()


def _overlapping(boxes: list[list]) -> list[str]:
    out = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if a[0] != b[0]:
                continue
            if min(a[4], b[4]) - max(a[2], b[2]) > 1 and min(a[5], b[5]) - max(a[3], b[3]) > 1:
                out.append(f"{a[0]}: {a[1]} on {b[1]}")
    return out


@pytest.mark.parametrize("width", [1000, 1440])
def test_at_half_a_screen_and_at_full_width_nothing_runs_off_or_overprints(served, width) -> None:
    url, browser, *_ = served
    page, errors = _page(browser, f"{url}?shown=second,first&open=output-e-x1,input-F1",
                         width=width)
    try:
        page.locator("button", has_text="Split").first.click()       # the stacked layout
        _settle(page, 1500)
        assert page.evaluate("document.documentElement.scrollWidth") <= width, "the page scrolls sideways"
        past = page.evaluate(_PAST_THE_EDGE)
        assert not past, f"past the window's right edge at {width} px: {past[:8]}"
        assert not _overlapping(page.evaluate(_WORD_BOXES)), _overlapping(page.evaluate(_WORD_BOXES))
        assert not errors, errors
    finally:
        page.close()


def test_the_system_form_shows_a_labelled_row_per_equation_and_adds_one(served) -> None:
    """The developer, 2026-11-11: rows for each state and each output equation, and θ, labelled,
    with add and remove. Switching the piece logs a plotly page error on its own, for every piece
    (found 2026-11-11, open in .claude/memory/project_state.md); the form must add none."""
    url, browser, *_ = served
    page, errors = _page(browser, url)
    try:
        page.locator("select").first.select_option(label="System")
        _settle(page, 2500)
        switched = len(errors)
        page.locator("button", has_text="New run").first.click()
        _settle(page, 1500)
        assert [t for t in _text(page) if t.endswith(" =")] == [
            "x1' =", "x2' =", "x3' =", "x4' =", "y1 =", "y2 =", "y3 =", "y4 =",
            "θ1 =", "θ2 =", "θ3 =", "θ4 ="]
        page.locator("button", has_text="+ State").first.click()
        _settle(page, 1500)
        assert "x5' =" in _text(page)
        x4 = "theta2*x1 + theta4*x2 - (theta1 + theta2)*x3 - (theta3 + theta4)*x4 + u2"
        box = next(b for b in page.locator("input[type=text]").all() if b.input_value() == x4)
        box.click()
        page.keyboard.press("End")
        page.keyboard.type(" - x5", delay=40)                   # key by key, as a person types
        _settle(page, 1500)
        assert box.input_value() == x4 + " - x5", "every key reached the box (code review)"
        assert not errors[switched:], errors[switched:]
    finally:
        page.close()
