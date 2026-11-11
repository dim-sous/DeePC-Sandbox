"""The page, driven in Python (BESS's ``tests/test_lab_view.py``, adapted).

A screen is validated by DRIVING it, never by a checklist (RCWS's own Lab incident: four designs
passed their checklists and were rejected on first contact). These tests drive the widgets a
person touches -- not the handlers behind them -- with ``launch``, ``capture``, ``every`` and
``toast`` injected, and two of them drive the real ``uv run deepc run`` and ``uv run deepc draw``
through the page end to end.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import socket
import time
import types
import urllib.request
from pathlib import Path
from typing import Any

import panel as pn
import pytest

from lab import jobs, skin, state
from lab.app import Lab, page, serve
import terminal
from tests.lab_runs import make_runs

ROOT = str(Path(__file__).resolve().parent.parent)
#: The closed loop's: it applies only while the noise-adaptive λ is on.
_W = "noise_estimation_window"


def _dataset(root: str, out_dir: str, name: str, *sets: str) -> str:
    """A saved dataset run under ``<root>/<out_dir>/dataset/``, written by the terminal; its path
    as the Lab offers it."""
    terminal.main(["run", "dataset", "--name", name, "--out-dir", str(Path(root) / out_dir),
                   *(w for item in sets for w in ("--set", item))])
    return f"{out_dir}/dataset/{name}.json"


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory) -> tuple[str, str]:
    return make_runs(tmp_path_factory.mktemp("lab-view"))


class _Ticker:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.stopped = 0

    def __call__(self, callback, period_ms):
        self.calls.append((callback, period_ms))
        return self

    def stop(self) -> None:
        self.stopped += 1


class _Process:
    def __init__(self) -> None:
        self.terminated = 0

    def terminate(self) -> None:
        self.terminated += 1


class _Launcher:
    """A ``launch`` that records the argv and hands back a job the test finishes by hand."""

    def __init__(self) -> None:
        self.argvs: list[list[str]] = []
        self.cwds: list[str] = []
        self.job: jobs.Job | None = None

    def __call__(self, argv, cwd=None) -> jobs.Job:
        self.argvs.append(list(argv))
        self.cwds.append(cwd)
        self.job = jobs.Job(list(argv), process=_Process())
        return self.job

    def finish(self, returncode: int, *lines: str) -> None:
        assert self.job is not None
        self.job.output.extend(line + "\n" for line in lines)
        self.job.returncode = returncode


class _Capture:
    """A ``capture`` that records the argv and answers as told, or raises."""

    def __init__(self, code: int = 0, output: str = "", raises: Exception | None = None) -> None:
        self.argvs: list[list[str]] = []
        self.code, self.output, self.raises = code, output, raises

    def __call__(self, argv, cwd=None) -> tuple[int, str]:
        self.argvs.append(list(argv))
        if self.raises is not None:
            raise self.raises
        return self.code, self.output


def _lab(two_runs, capture=None) -> tuple[Lab, _Launcher, _Ticker, list[str]]:
    root, out_dir = two_runs
    launcher, ticker, toasts = _Launcher(), _Ticker(), []
    lab = Lab(root, out_dir, launch=launcher, every=ticker, toast=toasts.append,
              capture=capture or _Capture(), schedule=lambda work: work())
    lab.page()
    return lab, launcher, ticker, toasts


def _type(lab: Lab, name: str, text: str) -> None:
    """Type into a box as a person does: the browser sends both of its values."""
    box = lab._controls[name]
    box.value = text
    box.value_input = text


def _pick(lab: Lab, name: str, value: str) -> None:
    lab._controls[name].value = value


def _name(lab: Lab, text: str) -> None:
    lab.run_name.value = text
    lab.run_name.value_input = text


def _sweep(lab: Lab, field: str, values: str) -> None:
    lab.sweep_field.value = field
    lab.sweep_values.value = values
    lab.sweep_values.value_input = values


def _names(lab: Lab) -> list[str]:
    """The run names the list shows, newest first."""
    return [row["name"] for row in lab.run_list.rows]


def _tick(lab: Lab, name: str, on: bool = True) -> None:
    """Click a row in the list: the component toggles the name in ``shown``."""
    shown = list(lab.run_list.shown)
    if on and name not in shown:
        shown.append(name)
    elif not on and name in shown:
        shown.remove(name)
    lab.run_list.shown = shown


def _tabs(lab: Lab) -> dict[str, pn.widgets.Toggle]:
    return lab.toggles()


def _card_of(lab: Lab, chart: str = "", number: str = ""):
    return next(c for c in lab.cards if (c.chart == chart if chart else c.number == number))


def _rows_of(lab: Lab, panel: str) -> list[str]:
    """The runs a chart card draws, in order."""
    return list(lab.drawn.get(_card_of(lab, panel).id, []))


def _card(lab: Lab, panel: str) -> pn.Column:
    for card in lab.cards_box.objects:
        top = getattr(card, "objects", [None])[0]
        title = getattr(top, "objects", [None])[0] if top is not None else None
        if f"data-panel='{html.escape(panel)}'" in getattr(title, "object", ""):
            return card
    raise AssertionError(f"no card for {panel}; cards: {lab.cards_box.objects}")


def _card_words(lab: Lab, panel: str) -> list[str]:
    """The lines a card carries beside or instead of its figure, as words: its description and
    its foot."""
    return [html.unescape(re.sub(r"<[^>]+>", "", o.object))
            for o in _card(lab, panel).objects[1:] if isinstance(o, pn.pane.HTML)]


def _greyed(lab: Lab, panel: str) -> dict[str, str]:
    """A card's greyed controls: their words -> the reason on their hover."""
    out = {}
    tools = next(o for o in _card(lab, panel).objects[1:] if isinstance(o, pn.FlexBox))
    for group in tools.objects:
        for o in group.objects:
            if isinstance(o, pn.pane.HTML) and "opacity:.45" in o.object:
                words = html.unescape(re.sub(r"<[^>]+>", "", o.object))
                out[words] = html.unescape(re.search(r'title="([^"]*)"', o.object).group(1))
    return out


def _control(lab: Lab, panel: str, words: str) -> pn.widgets.Toggle:
    tools = next(o for o in _card(lab, panel).objects[1:] if isinstance(o, pn.FlexBox))
    return next(o for g in tools.objects for o in g.objects
                if isinstance(o, pn.widgets.Toggle) and o.name == words)


def _save_button(lab: Lab, panel: str) -> pn.widgets.Button:
    top = _card(lab, panel).objects[0]
    return next(o for o in top.objects if isinstance(o, pn.widgets.Button) and o.name == "Save")


def _columns(lab: Lab) -> list[str]:
    """The compare table's column heads: one per shown run, in the order they were ticked."""
    return list(lab.compare_table.heads)


def _console(lab: Lab) -> str:
    return lab.console_text.object


# ---------------------------------------------------------------- running a piece

def test_run_launches_the_form_as_the_command_and_shows_what_it_wrote(two_runs, capsys) -> None:
    lab, launcher, ticker, toasts = _lab(two_runs)
    root, out_dir = two_runs
    assert "Nothing has been run in this session." in _console(lab)
    assert _names(lab) == ["second", "first"]
    assert not lab.console.visible and not lab.drawer.visible

    lab.open_drawer()
    _type(lab, "lambda_g", "10")
    _name(lab, "probe")
    expected = jobs.run_argv("closed_loop", ["lambda_g=10"], "probe", out_dir)
    assert jobs.line_of(expected) in lab.command_line.object, "the line Run would type, live"
    lab._run()
    assert launcher.argvs == [expected] and launcher.cwds == [root]
    assert capsys.readouterr().out.strip() == f"lab: {jobs.line_of(expected)}"
    assert ticker.calls and ticker.calls[0][1] == 700
    assert lab.run_btn.disabled and not lab.stop_btn.disabled and "Running" in lab.job_tag.object
    assert lab.console.visible and not lab.drawer.visible, "Run opens the terminal, shuts the form"
    assert f"lab: {jobs.line_of(expected)}" in _console(lab)

    # the terminal writes the record and prints its path; the page lists it, ticked
    written = Path(root) / out_dir / "closed_loop" / "probe.json"
    shutil.copy(Path(root) / out_dir / "closed_loop" / "first.json", written)
    launcher.finish(0, f"{out_dir}/closed_loop/probe.json", "  rmse_position = 0.01")
    lab._poll()
    assert ticker.stopped == 1 and lab._ticker is None
    assert _names(lab)[0] == "probe" and "probe.json" in _console(lab)
    assert "probe" in lab.shown and "probe" in _columns(lab)
    assert not lab.run_btn.disabled and lab.stop_btn.disabled and "Done" in lab.job_tag.object
    assert "wrote 1 record" in lab.last_line.object
    assert toasts[-1] == "Closed loop: wrote 1 record"


def test_a_running_job_greys_run_and_a_second_click_launches_nothing(two_runs) -> None:
    lab, launcher, _ticker, toasts = _lab(two_runs)
    _name(lab, "a b")
    assert "cannot name a record" in lab.command_line.object, "said before Run is pressed"
    lab._run()
    assert launcher.argvs == [] and lab.banner.visible and "cannot name a record" in lab.banner.object

    _name(lab, "ok")
    lab._run()
    assert len(launcher.argvs) == 1 and not lab.banner.visible
    lab._run()
    assert len(launcher.argvs) == 1 and toasts[-1] == "A run is in progress — stop it first"


@pytest.mark.parametrize("stop,returncode,lines,words", [
    (False, 1, (), "the run was refused"),
    (True, 143, (), "stopped after it wrote 0 records"),
    (False, 1, ("RECORD",), "wrote 1 record, then ended with exit 1"),
])
def test_what_a_finished_run_did_counts_what_it_wrote_and_not_only_how_it_ended(
        two_runs, stop, returncode, lines, words) -> None:
    """A stopped run refused nothing, and a sweep can write some records before being refused."""
    lab, launcher, _ticker, toasts = _lab(two_runs)
    root, out_dir = two_runs
    _name(lab, "outcome")
    lab._run()
    if stop:
        lab._stop()
        assert launcher.job.process.terminated == 1 and toasts[-1] == "Stopping…"
    printed = []
    for line in lines:
        if line == "RECORD":
            target = Path(root) / out_dir / "closed_loop" / "part.json"
            shutil.copy(Path(root) / out_dir / "closed_loop" / "first.json", target)
            printed.append(f"{out_dir}/closed_loop/part.json")
    launcher.finish(returncode, *printed)
    lab._poll()
    assert toasts[-1] == f"Closed loop: {words}"
    assert not lab.run_btn.disabled


def test_a_refusal_from_the_terminal_lands_in_the_console_in_its_own_words(two_runs) -> None:
    lab, launcher, _ticker, toasts = _lab(two_runs)
    _name(lab, "bad")
    lab._run()
    launcher.finish(1, "deepc: Ts: 'x' is not a number")
    lab._poll()
    assert "is not a number" in _console(lab)
    assert toasts[-1] == "Closed loop: the run was refused" and "exit 1" in lab.job_tag.object


def test_the_output_of_one_run_stays_in_the_terminal_under_the_next(two_runs) -> None:
    lab, launcher, _ticker, _toasts = _lab(two_runs)
    _name(lab, "one")
    lab._run()
    launcher.finish(0, "one said this")
    lab._poll()
    _name(lab, "two")
    lab._run()
    launcher.finish(0, "two said that")
    lab._poll()
    text = _console(lab)
    assert text.index("one said this") < text.index("--name two") < text.index("two said that")
    assert text.count("one said this") == 1, "folded into the history once, not on every redraw"


def test_reading_what_a_run_wrote_shows_its_failure_on_the_page(two_runs, monkeypatch) -> None:
    """Panel's periodic callback has no exception hook, so the poll routes its own failures."""
    lab, launcher, _ticker, _toasts = _lab(two_runs)
    _name(lab, "boom")
    lab._run()
    launcher.finish(0)

    def raises() -> None:
        raise ValueError("the out-dir went away")

    monkeypatch.setattr(lab, "reload", raises)
    lab._poll()                                      # does not propagate
    assert lab.banner.visible and "the out-dir went away" in lab.banner.object


def test_use_loads_a_runs_settings_into_the_form_and_the_widgets_and_clear_empties(two_runs) -> None:
    lab, _launcher, _ticker, toasts = _lab(two_runs)
    lab.open_drawer("second")                        # what the list's `use` reaches
    assert lab.drawer.visible and "From second" in lab.from_words.object
    data = str(Path(two_runs[0]) / two_runs[1] / "dataset" / "default.json")
    assert lab.form.set == {"dataset_run": data, "sim_duration": "1", "lambda_g": "50"}
    assert lab._controls["lambda_g"].value == "50" and lab._controls["dataset_run"].value == data
    assert lab._controls["N"].value == ""
    assert skin.CHANGED in lab._controls["lambda_g"].stylesheets, "a value away from its default"
    assert skin.CHANGED not in lab._controls["N"].stylesheets
    assert "1 of 9 changed" in lab._group_notes["Controller"].object
    assert toasts[-1] == "3 settings loaded from second"
    lab._clear()
    assert lab.form.set == {} and lab._controls["lambda_g"].value == ""
    assert skin.CHANGED not in lab._controls["lambda_g"].stylesheets
    assert "From defaults" in lab.from_words.object
    lab.close_drawer()
    assert not lab.drawer.visible


def test_a_value_the_page_itself_wrote_does_not_hide_the_next_keystroke(two_runs) -> None:
    """A box carries two values and a programmatic write sets one: retyping what ``use`` loaded
    over must reach the form, or the run uses a number the box does not show."""
    lab, _launcher, _ticker, _toasts = _lab(two_runs)
    _type(lab, "lambda_g", "50")
    lab._use("second")                               # writes lambda_g = 50 over the typed 50
    _type(lab, "lambda_g", "5")
    assert lab.form.set["lambda_g"] == "5"
    overrides = lab.form.launch_words()[0]
    assert "lambda_g=5" in overrides and "lambda_g=50" not in overrides


def test_a_sweep_the_page_cleared_is_not_resurrected_by_picking_an_axis(two_runs) -> None:
    lab, _launcher, _ticker, _toasts = _lab(two_runs)
    _sweep(lab, "lambda_g", "1,5")
    assert lab.form.launch_words()[1] == ["lambda_g=1,5"]
    lab._clear()
    lab.sweep_field.value = "N"                      # an axis, with the values box now empty
    assert lab.form.launch_words() == ([], []) and lab.form.sweep_values == ""


def test_the_dataset_run_is_picked_from_the_dataset_pieces_saved_runs(own_runs) -> None:
    """RCWS's record picker (``record_of``): the closed loop's form offers the dataset piece's
    saved runs by name, the default first, and launches the path picked."""
    lab, launcher, *_ = _lab(own_runs)
    root, out_dir = own_runs
    _dataset(root, out_dir, "default")
    noisy = _dataset(root, out_dir, "noisy", "noise_std_output=0.05")
    lab.reload()
    assert lab._controls["dataset_run"].options == {"default (default)": "", "noisy": noisy}
    _pick(lab, "dataset_run", noisy)
    _name(lab, "on-noisy")
    lab._run()
    assert launcher.argvs[-1] == jobs.run_argv("closed_loop", [f"dataset_run={noisy}"],
                                               "on-noisy", out_dir)


def test_a_features_settings_are_folded_until_opened_and_say_what_changed(two_runs) -> None:
    """The noise-adaptive λ's four settings, folded: the heading opens them, its note counts what
    changed, and a redraw keeps them open."""
    lab, *_ = _lab(two_runs)

    def head_and_body():
        for section in lab.form_box.objects:
            head, body = section.objects[0].objects[0], section.objects[1]
            if isinstance(head, pn.widgets.Toggle) and head.label.endswith("Noise-adaptive"):
                return head, body
        raise AssertionError("no folded group for the noise-adaptive λ's settings")

    head, body = head_and_body()
    assert head.label == "▸ Noise-adaptive" and not body.visible
    assert "4 at defaults" in lab._group_notes["Noise-adaptive"].object
    head.value = True
    assert head.label == "▾ Noise-adaptive" and body.visible
    _pick(lab, "noise_adaptive", "true")
    _type(lab, _W, "10")
    assert "1 of 4 changed" in lab._group_notes["Noise-adaptive"].object
    lab._draw_form()
    head, body = head_and_body()
    assert body.visible and head.value, "opened, it stays open"


def test_a_typed_box_is_empty_and_greyed_once_a_pick_shuts_its_gate(two_runs) -> None:
    lab, *_ = _lab(two_runs)
    _pick(lab, "noise_adaptive", "true")             # W applies while the feature is on only
    _type(lab, _W, "10")
    assert lab._controls[_W].value == "10" and lab.form.set == {"noise_adaptive": "true", _W: "10"}
    _pick(lab, "noise_adaptive", "")
    box = lab._controls[_W]
    assert box.value == "" and box.value_input == "" and box.disabled
    assert lab.form.set == {}
    assert "applies only when Noise-adaptive λ is true" in lab._labels[_W].object
    assert "taken back" in lab.form_note.object      # said, not silently dropped
    _pick(lab, "noise_adaptive", "true")
    assert not box.disabled


def test_the_sweep_axis_box_is_greyed_once_the_sweep_row_names_it_and_holds_values(two_runs) -> None:
    lab, launcher, *_ = _lab(two_runs)
    _type(lab, "lambda_g", "10")
    lab.sweep_field.value = "lambda_g"
    assert not lab._controls["lambda_g"].disabled           # named, no values yet
    _sweep(lab, "lambda_g", "1,5")
    assert lab._controls["lambda_g"].disabled and lab._controls["lambda_g"].value == ""
    assert "is the sweep axis" in lab._labels["lambda_g"].object
    _name(lab, "sw")
    lab._run()
    assert launcher.argvs[-1] == jobs.run_argv("closed_loop", [], "sw", two_runs[1],
                                               ["lambda_g=1,5"])


def test_a_record_that_cannot_be_read_is_listed_and_offers_nothing_to_load(two_runs, tmp_path) -> None:
    root, out_dir = two_runs
    folder = tmp_path / out_dir / "closed_loop"
    folder.mkdir(parents=True)
    shutil.copy(Path(root) / out_dir / "closed_loop" / "first.json", folder / "good.json")
    (folder / "kept-setup.json").write_text('{"lambda_g": 2}')
    lab = Lab(str(tmp_path), out_dir, every=_Ticker(), toast=lambda _w: None)
    lab.page()
    rows = {row["name"]: row for row in lab.run_list.rows}
    assert sorted(rows) == ["good", "kept-setup"]
    assert not rows["good"]["stale"]
    assert "carries no" in rows["kept-setup"]["stale"], "a stale row says why under its name"
    assert lab.shown == ["good"]


def test_a_page_that_cannot_be_built_shows_why_instead_of_nothing(tmp_path: Path) -> None:
    lab = Lab(str(tmp_path), "results", every=_Ticker(), toast=lambda _w: None)
    assert isinstance(lab.page(), pn.Column)               # the healthy case, for contrast

    def raises(*_a, **_k):
        raise ValueError("the registry would not read")

    import lab.app as app_module
    original = app_module.Lab
    app_module.Lab = raises
    try:
        shown = page(str(tmp_path), "results")
    finally:
        app_module.Lab = original
    assert isinstance(shown, pn.pane.HTML)
    assert "could not build its page" in shown.object and "would not read" in shown.object


def test_the_page_is_served_on_127_0_0_1_and_builds_for_that_root(two_runs) -> None:
    root, out_dir = two_runs
    assert isinstance(page(root, out_dir), pn.Column), "a traceback pane would also carry a title"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    handler = pn.config.exception_handler
    server = serve(port, root, out_dir, show=False, threaded=True)
    try:
        deadline = time.time() + 20
        while True:
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.2)
        assert "<title>DeePC Lab</title>" in body.decode()
    finally:
        server.stop()
        pn.config.exception_handler = handler          # `serve` sets it process-wide


def test_a_real_run_from_the_page_writes_a_record_the_list_shows(tmp_path: Path, capsys) -> None:
    """End to end: the real ``uv run deepc run`` subprocess, launched by the page from the project
    root into a temporary out-dir, and stopped by the page if it outlives the wait."""
    out_dir = str(tmp_path / "results")
    toasts: list[str] = []
    data = _dataset("/", out_dir, "short")           # what the closed loop runs on
    capsys.readouterr()                              # the terminal's own lines for it
    lab = Lab(ROOT, out_dir, every=_Ticker(), toast=toasts.append)
    _pick(lab, "dataset_run", data)
    _type(lab, "sim_duration", "1")
    _name(lab, "live")
    lab._run()
    assert lab.job is not None and f"--out-dir {out_dir}" in lab.job.line
    deadline = time.time() + 180
    while lab.job.running and time.time() < deadline:
        time.sleep(0.3)
    if lab.job.running:
        lab.stop_job()
        pytest.fail(f"the run did not finish in 180 s: {''.join(lab.job.output)}")
    lab._poll()
    assert lab.job.returncode == 0, "".join(lab.job.output)
    record = Path(out_dir) / "closed_loop" / "live.json"
    assert record.exists() and json.loads(record.read_text())["config"]["sim_duration"] == 1
    assert _names(lab) == ["live"] and str(record) in _console(lab)
    assert lab.shown == ["live"] and _rows_of(lab, "output-e-lateral") == ["live"], \
        "shown, and charted"
    assert toasts[-1] == "Closed loop: wrote 1 record"
    assert capsys.readouterr().out.startswith(
        f"lab: uv run deepc run closed_loop --set dataset_run={data}")


def test_the_end_of_a_browser_session_stops_the_run_it_left_going(two_runs) -> None:
    """A closed tab otherwise leaves a full run burning CPU with no page watching it."""
    lab, launcher, _ticker, _toasts = _lab(two_runs)
    _name(lab, "orphan")
    lab._run()
    assert lab.stop_job() is True and launcher.job.process.terminated == 1
    launcher.job.returncode = -15
    assert lab.stop_job() is False


def test_a_running_job_is_reachable_with_the_form_open(two_runs) -> None:
    """Pressing stop must not need the drawer the Run button is in."""
    lab, _launcher, _ticker, _toasts = _lab(two_runs)
    lab.open_drawer()
    _name(lab, "going")
    lab._run()
    assert not lab.drawer.visible and not lab.stop_btn.disabled and "Running" in lab.job_tag.object
    lab.open_drawer()
    assert lab.drawer.visible and not lab.stop_btn.disabled, "stop is in the header, not the drawer"


# ---------------------------------------------------------------- the runs and their charts

@pytest.fixture
def own_runs(two_runs, tmp_path) -> tuple[str, str]:
    """A private copy of the two records: these tests write files beside them and re-stamp
    them, and the shared fixture's folder is read by every other test in this file."""
    root, out_dir = two_runs
    src, dst = Path(root) / out_dir / "closed_loop", tmp_path / out_dir / "closed_loop"
    dst.mkdir(parents=True)
    for name in ("first", "second"):
        shutil.copy(src / f"{name}.json", dst / f"{name}.json")
    now = time.time()
    os.utime(dst / "first.json", (now - 60, now - 60))     # second is the newer one
    os.utime(dst / "second.json", (now, now))
    return str(tmp_path), out_dir


def test_one_browser_message_carrying_both_values_of_a_box_is_one_edit(own_runs) -> None:
    """A paste, or a key and Tab together, reach the page as ONE message holding the box's two
    values. One watcher over both got two events from it and the run kept its default while the
    box showed the value (2026-10-27, headless Chromium): the box is read on `value_input`."""
    lab, *_ = _lab(own_runs)
    lab._controls["lambda_g"].param.update(value="4", value_input="4")
    assert lab.form.set == {"lambda_g": "4"} and "--set lambda_g=4" in lab.command_line.object
    lab.run_name.param.update(value="probe", value_input="probe")
    assert lab.form.name == "probe"
    lab.sweep_values.param.update(value="1,2", value_input="1,2")
    assert lab.form.sweep_values == "1,2"


def test_the_newest_run_is_shown_and_charted_and_ticking_another_adds_it_and_a_column(own_runs) -> None:
    lab, *_ = _lab(own_runs)
    assert lab.shown == ["second"], "the newest, so the screen is never empty"
    assert lab.open_charts == ["output-e-lateral"] and _rows_of(lab, "output-e-lateral") == ["second"], \
        "and its first panel open"
    assert _columns(lab) == ["second"]
    assert "tick another run" in lab.compare_count.object
    _tick(lab, "first")
    assert lab.shown == ["second", "first"]
    assert _columns(lab) == ["second", "first"], "the columns follow the order ticked"
    assert _rows_of(lab, "output-e-lateral") == ["second", "first"], "every shown run, in that order"
    assert "1 setting" in lab.compare_count.object and "differ" in lab.compare_count.object
    colours = [r["colour"] for r in lab.run_list.rows]
    assert colours[0] and colours[1] and colours[0] != colours[1], "a shown run's dot is its colour"
    _tick(lab, "second", False)
    assert lab.shown == ["first"] and _columns(lab) == ["first"] and _rows_of(lab, "output-e-lateral") == ["first"]


def test_opening_a_chart_is_the_figure_layers_and_asks_the_terminal_for_nothing(own_runs) -> None:
    capture = _Capture()
    lab, *_ = _lab(own_runs, capture)
    _tabs(lab)["solve-time"].value = True
    assert capture.argvs == [], "a chart is drawn by the figure layer in the page, not a subprocess"
    assert [lab.cards_box.objects.index(_card(lab, p))
            for p in ("output-e-lateral", "solve-time")] == [0, 1], "cards in the record's declared order"
    fig = lab._panes[_card_of(lab, "solve-time").id][0].object
    assert fig["layout"]["yaxis"]["title"]["text"] == "Solve time [s]", "the record's own words"
    assert fig["layout"]["xaxis"]["title"]["text"] == "Solve [-]"
    title = _card(lab, "solve-time").objects[0].objects[0].object
    assert "Wall-clock time of the solve" in html.unescape(title), "its description on the title's hover"
    assert not any("Wall-clock time" in w for w in _card_words(lab, "solve-time")), \
        "and not repeated under it (RCWS)"
    assert "2 of 12 open" in lab.open_count.object
    assert list(lab._lines) == ["Outputs", "Inputs", "Solver", "Position"], \
        "listed in the record's groups"
    assert all(f">{group}<" in line.objects[0].object for group, line in lab._lines.items())
    _tabs(lab)["output-e-lateral"].value = False
    assert lab.open_charts == ["solve-time"]


def test_a_control_a_card_cannot_use_is_greyed_with_the_reason_on_its_hover(own_runs) -> None:
    lab, *_ = _lab(own_runs)
    greyed = _greyed(lab, "output-e-lateral")
    assert greyed["Run | Quantity"] == "One run shown"
    assert greyed["Overlay | Split | Mean | RMS"] == "One run shown"
    assert greyed["Log x"] == "Its bottom axis holds 1 value(s) at or below zero"
    _tick(lab, "first")
    greyed = _greyed(lab, "output-e-lateral")
    assert greyed["Mean"] == "The chart declares no statistic", \
        "the figure layer's own words, begun with a capital"
    assert greyed["RMS"] == "The chart declares no statistic"
    assert "Run | Quantity" not in greyed


def test_runs_that_differ_in_their_seed_alone_pool_on_a_card_into_one_line(own_runs) -> None:
    """The dataset's charts declare Mean and RMS; two seeds of one dataset run pool into one line
    per quantity. The closed loop holds no seed, so its charts declare no statistic."""
    root, out_dir = own_runs
    for seed in (1, 2):
        _dataset(root, out_dir, f"seed-{seed}", f"seed={seed}")
    lab, *_ = _lab(own_runs)
    lab._on_piece("dataset")
    _tick(lab, "seed-1")
    _tabs(lab)["output-e-lateral"].value = True
    assert "Mean" not in _greyed(lab, "output-e-lateral")
    _control(lab, "output-e-lateral", "Mean").value = True
    fig = lab._panes[_card_of(lab, "output-e-lateral").id][0].object
    names = [t["name"] for t in fig["data"] if t.get("showlegend") is not False]
    assert names == ["Mean of 2 runs · e_lateral"]
    assert _greyed(lab, "output-e-lateral")["Run | Quantity"] == \
        "The pooled runs are one line per quantity"


def test_a_table_in_a_cards_place_is_the_numbers_on_its_chart(own_runs) -> None:
    lab, *_ = _lab(own_runs)
    _tabs(lab)["position-x"].value = True
    _control(lab, "position-x", "Table").value = True
    record = lab._record_of(lab.shown_runs()[0])[0]
    position = next(s for s in record.series() if s.name == "position-x")
    body = next(o for o in _card(lab, "position-x").objects if isinstance(o, pn.pane.HTML)
                and "<table>" in o.object)
    assert "<th>Plant</th>" in body.object and "<th>Reference</th>" in body.object
    assert f"<td>{float(position.traces[0][1][0]):.6g}</td>" in body.object
    assert _card_of(lab, "position-x").id not in lab._panes, "a table in the chart's place"


def test_a_number_clicked_in_the_compare_table_is_a_card_every_run_a_point(own_runs) -> None:
    lab, *_ = _lab(own_runs)
    _tick(lab, "first")
    lab.compare_table._handle_msg({"number": "rmse_position"})
    card = _card_of(lab, number="rmse_position")
    assert card.x == "lambda_g", "the setting that varies is the bottom axis"
    assert lab.drawn[card.id] == ["second", "first"]
    title = _card(lab, "rmse_position")
    assert title is not None
    assert next(r for r in lab.compare_table.rows if r["key"] == "rmse_position")["open"]
    greyed = _greyed(lab, "rmse_position")
    assert greyed["All | Mean | RMS"] == "No colour holds two runs at one value"
    lab.compare_table._handle_msg({"number": "rmse_position"})
    assert not [c for c in lab.cards if c.kind == "numbers"], "asked again, it closes"


def test_save_draws_the_card_exactly_as_shown_and_the_folder_is_listed(own_runs) -> None:
    root, out_dir = own_runs
    capture = _Capture(output=f"{out_dir}/closed_loop/second+first/output-e-lateral.png\n")
    lab, *_ = _lab(own_runs, capture)
    _tick(lab, "first")
    _control(lab, "output-e-lateral", "Split").value = True
    lab._panes[_card_of(lab, "output-e-lateral").id][0].relayout_data = {"xaxis.range[0]": 0.2,
                                                                         "xaxis.range[1]": 0.8}
    _save_button(lab, "output-e-lateral").clicks += 1
    [argv] = capture.argvs
    assert argv[:6] == ["uv", "run", "deepc", "draw",
                        os.path.join(out_dir, "closed_loop", "second.json"),
                        os.path.join(out_dir, "closed_loop", "first.json")]
    words = " ".join(argv[6:])
    assert "--panel output-e-lateral" in words and "--runs rows" in words \
        and "--x-range=0.2,0.8" in words
    assert "--colour-by" not in words, "a greyed control travels as nothing"
    assert f"{out_dir}/closed_loop/second+first/output-e-lateral.png" in _console(lab)
    assert "Wrote" in _card_words(lab, "output-e-lateral")[-1], \
        "the card's foot says what its save wrote"
    assert lab.console.visible and "exit 0" in lab.last_line.object


def test_a_refused_draw_says_so_in_the_terminal_and_is_not_latched(own_runs) -> None:
    capture = _Capture(raises=ValueError("a command is in progress: uv run deepc run closed_loop"))
    lab, *_ = _lab(own_runs, capture)
    _save_button(lab, "output-e-lateral").clicks += 1
    assert "refused: a command is in progress" in _console(lab)
    capture.raises = None
    _save_button(lab, "output-e-lateral").clicks += 1
    assert len(capture.argvs) == 2, "asked again: the reason it was refused had passed"


def test_a_draw_that_cannot_start_leaves_the_page_readable(own_runs) -> None:
    capture = _Capture(raises=FileNotFoundError(2, "No such file or directory", "uv"))
    lab, *_ = _lab(own_runs, capture)
    _save_button(lab, "output-e-lateral").clicks += 1
    assert "could not be started" in _console(lab) and "FileNotFoundError" in _console(lab)
    assert lab.page().get_root() is not None


def _on_show(lab: Lab) -> list[str]:
    """The groups whose lines the chart list shows below its index, in its order."""
    return [group for group, line in lab._lines.items() if group and line.visible]


def test_the_chart_list_opens_on_an_index_of_its_groups(own_runs, monkeypatch) -> None:
    """RCWS #255, the developer's: above the lines, the groups by name, read down three columns,
    each name a toggle that shows its group's lines -- 45 charts in 15 groups, every group shown,
    put the first chart below the bottom of a 1700 x 1100 window. A chart opened shows its group,
    by its toggle, the address or as the first chart a page opens on; hiding a group closes
    nothing and stays; show all and hide all; a list with no groups has no index."""
    lab, *_ = _lab(own_runs)
    named = ["Outputs", "Inputs", "Solver", "Position"]
    index = lab.tabs.objects[0]
    columns = [[toggle.label for toggle in column.objects] for column in index.objects]
    assert len(columns) == 3 and [label[2:] for column in columns for label in column] == named
    assert list(lab.group_toggles()) == named and "output-e-lateral" in lab.toggles()
    assert _on_show(lab) == ["Outputs"]                   # the first chart a page opens on
    assert lab.group_toggles()["Outputs"].label == "▾ Outputs"
    assert lab.group_toggles()["Solver"].label == "▸ Solver"
    lab.group_toggles()["Solver"].value = True
    assert _on_show(lab) == ["Outputs", "Solver"]
    lab.group_toggles()["Solver"].value = False
    assert _on_show(lab) == ["Outputs"]
    _tabs(lab)["cost"].value = True                       # a hidden group's toggle is still there
    assert _on_show(lab) == ["Outputs", "Solver"]
    lab.group_toggles()["Solver"].value = False
    _tick(lab, "first")                                   # a redraw keeps the developer's choice
    assert _on_show(lab) == ["Outputs"] and "cost" in lab.open_charts, "hiding closes nothing"
    assert lab.show_all.visible and lab.hide_all.visible
    lab.show_all.clicks += 1
    assert _on_show(lab) == named
    lab.hide_all.clicks += 1
    assert _on_show(lab) == []
    # the address's charts show their groups
    location = types.SimpleNamespace(query_params={"shown": "first", "open": "cost"},
                                     update_query=lambda **query: None)
    monkeypatch.setattr(type(pn.state), "location", property(lambda state: location))
    linked, *_ = _lab(own_runs)
    assert linked.open_charts == ["cost"] and _on_show(linked) == ["Solver"]
    # charts in no group: no index, and nothing to show or hide
    monkeypatch.setattr(lab, "_drawable", lambda run: {**run, "panel_groups": {}})
    lab._draw_tabs()
    assert not lab.group_toggles() and not lab._lines and not lab.show_all.visible
    assert "output-e-lateral" in lab.toggles()


def test_the_chart_list_is_what_the_records_class_declares_not_what_its_file_lists(own_runs) -> None:
    """A run's file lists the charts its class declared when it was saved. After a chart change
    that list is behind: read off the file, the Lab offered charts the class no longer draws and
    none it now does, until every run was saved again (RCWS #257). The list is what the class
    declares now -- and a chart of it opens."""
    root, out_dir = own_runs
    path = Path(root) / out_dir / "closed_loop" / "second.json"
    raw = json.loads(path.read_text())
    raw["panels"] = ["output-e-lateral", "a-chart-no-longer-declared"]
    path.write_text(json.dumps(raw))
    lab, *_ = _lab(own_runs)
    assert lab.shown == ["second"]
    record = lab._record_of(next(r for r in lab.runs() if r["name"] == "second"))[0]
    assert list(lab.toggles()) == [s.name for s in record.series()]
    assert "a-chart-no-longer-declared" not in lab.toggles()
    _tabs(lab)["cost"].value = True
    assert _rows_of(lab, "cost") == ["second"]


def test_a_refused_record_is_read_once_per_version_of_its_file(own_runs, monkeypatch) -> None:
    """Read again at every redraw, a run the class refuses cost a read and a parse per click
    (code review, 2026-10-27): the refusal is kept with the version of the file it was for."""
    lab, *_ = _lab(own_runs)
    calls: list[str] = []

    def refuse(run):
        calls.append(run["name"])
        raise ValueError("refused in the class's own words")

    monkeypatch.setattr(state, "load", refuse)
    lab._loaded.clear()
    run = lab.runs()[0]
    assert lab._record_of(run) == (None, "refused in the class's own words")
    assert lab._record_of(run) == (None, "refused in the class's own words")
    assert calls == [run["name"]]


def test_the_header_counts_the_pieces_own_records(own_runs) -> None:
    """With two pieces, the count beside a piece's folder was every piece's (2026-10-27)."""
    root, out_dir = own_runs
    _dataset(root, out_dir, "default")
    lab, *_ = _lab(own_runs)
    assert lab.piece["name"] == "closed_loop" and "· 2 records" in lab.where.object


def test_a_click_on_the_chart_list_builds_none_of_it_again(own_runs) -> None:
    """RCWS #256: the toggles are built again only when what the list holds changes. Built again
    on every click, a click landing while the page drew the new ones was lost -- 2 of 5 charts
    clicked 80 ms apart in headless Chromium (2026-10-27). A click on the index, on a chart, on
    show all or hide all, or a card closed, sets the toggles already there."""
    lab, *_ = _lab(own_runs)
    on_screen, toggles, index = list(lab.tabs.objects), lab.toggles(), lab.group_toggles()
    lab.group_toggles()["Solver"].value = True
    _tabs(lab)["cost"].value = True
    assert lab.toggles()["cost"].value and lab.group_toggles()["Solver"].label == "▾ Solver"
    lab.show_all.clicks += 1
    lab.hide_all.clicks += 1
    lab._close(_card_of(lab, "cost"))
    assert len(lab.tabs.objects) == len(on_screen)
    assert all(now is before for now, before in zip(lab.tabs.objects, on_screen))
    assert all(lab.toggles()[name] is toggle for name, toggle in toggles.items())
    assert all(lab.group_toggles()[group] is toggle for group, toggle in index.items())
    assert not lab.toggles()["cost"].value
    assert lab.group_toggles()["Solver"].label == "▸ Solver"
    assert "1 of 12 open" in lab.open_count.object


def test_a_run_without_the_panel_is_the_figure_layers_refusal_on_the_card(own_runs) -> None:
    """Drawn over the runs that hold it, the chart would leave the others out unsaid; the figure
    layer refuses (the charts branch's rule), and the card says so in its words."""
    lab, *_ = _lab(own_runs)
    # `first` is made to declare its lateral error only, as an older record of the piece would
    real = lab._record_of
    lab._record_of = lambda run: ((types_ns(real(run)[0]), "") if run["name"] == "first"
                                  else real(run))
    for row in lab.records:
        if row["name"] == "first":
            row["panels"] = ["output-e-lateral"]
    _tick(lab, "first")
    _tabs(lab)["cost"].value = True
    assert _rows_of(lab, "cost") == [] and _card_of(lab, "cost").id not in lab._panes
    assert any("First declare no panel 'cost'" in w for w in _card_words(lab, "cost"))
    subs = [o.object for o in lab.tabs.select(pn.pane.HTML)]
    assert any("1 of 2 runs" in s for s in subs)


def types_ns(record):
    """``record`` declaring its lateral-error panel alone, a record of the same piece (the figure
    layer refuses to compare runs of two pieces)."""
    keep = [s for s in record.series() if s.name == "output-e-lateral"]
    older = type("Older", (), {"component": type(record).component})()
    for name, value in (("series", lambda: tuple(keep)), ("config", record.config),
                        ("numbers", record.numbers), ("number_specs", record.number_specs),
                        ("constants", record.constants)):
        setattr(older, name, value)
    return older


def test_a_record_that_does_not_load_is_a_line_on_the_card_in_the_loaders_words(own_runs) -> None:
    """A header the listing reads and a body the class refuses: listed, tickable, and the card
    says why in the class's own words rather than guessing."""
    root, out_dir = own_runs
    folder = Path(root) / out_dir / "closed_loop"
    raw = json.loads((folder / "first.json").read_text())
    raw["time"] = "oops"
    (folder / "zz-odd-body.json").write_text(json.dumps(raw))
    # stamped older than `second`, which stays the run shown first: left at its write time it
    # was the newest in a cloud container (2026-10-27), shown first, and the columns read
    # ['zz-odd-body']
    newest = (folder / "second.json").stat().st_mtime
    os.utime(folder / "zz-odd-body.json", (newest - 30, newest - 30))
    lab, *_ = _lab(own_runs)
    assert "zz-odd-body" in _names(lab)
    _tick(lab, "zz-odd-body")
    assert _columns(lab) == ["second", "zz-odd-body"], "its header reads; it compares"
    assert _rows_of(lab, "output-e-lateral") == ["second"]
    words = " ".join(_card_words(lab, "output-e-lateral"))
    assert "zz-odd-body" in words and "does not load as the current Record" in words


def test_the_settings_are_shown_when_asked_and_a_number_never_hidden(own_runs) -> None:
    lab, *_ = _lab(own_runs)
    _tick(lab, "first")
    assert lab.settings_choice.value == "none", "the numbers alone until asked"
    kinds = lambda: [r["kind"] for r in lab.compare_table.rows]
    assert "setting" not in kinds() and "number" in kinds()
    lab.settings_choice.value = "differ"
    differ = [r["term"] for r in lab.compare_table.rows if r["kind"] == "setting"]
    lab.settings_choice.value = "all"
    every = [r["term"] for r in lab.compare_table.rows if r["kind"] == "setting"]
    assert len(differ) == 1 and len(every) > len(differ)
    assert "Position RMSE [m]" in [r["term"] for r in lab.compare_table.rows]


def test_the_constants_a_run_used_are_compared_too(own_runs) -> None:
    """What a run used beyond its Config -- what independent truth needs when a
    number is judged -- read off the loaded record."""
    lab, *_ = _lab(own_runs)
    _tick(lab, "first")
    lab.settings_choice.value = "all"
    terms = [r["term"] for r in lab.compare_table.rows]
    assert "Constants" in terms and "deepc_config · lambda_g" in terms, "the value the code resolved"
    assert terms.index("Position RMSE [m]") < terms.index("Constants"), \
        "numbers first, constants last"


def test_the_filter_lists_the_runs_holding_every_word_and_all_and_none_tick_them(own_runs) -> None:
    lab, *_ = _lab(own_runs)
    lab.run_filter.value_input = "fir"
    assert _names(lab) == ["first"] and "1 listed" in lab.runs_count.object
    lab.tick_listed()
    assert lab.shown == ["second", "first"]
    lab.untick_listed()
    assert lab.shown == ["second"], "a run the filter does not list keeps its tick"


def test_deleting_a_run_asks_first_and_then_takes_only_its_record(own_runs) -> None:
    root, out_dir = own_runs
    lab, *_ = _lab(own_runs)
    lab.run_list._handle_msg({"delete": "first"})
    assert "Delete results/closed_loop/first.json?" in lab.question.object
    lab._on_no()
    assert (Path(root) / out_dir / "closed_loop" / "first.json").exists(), "cancel deletes nothing"
    lab.run_list._handle_msg({"delete": "first"})
    lab._on_yes()
    assert not (Path(root) / out_dir / "closed_loop" / "first.json").exists()
    assert _names(lab) == ["second"] and "lab: deleted" in _console(lab)
    lab._on_yes()                                    # a second yes finds nothing left to do
    assert _names(lab) == ["second"]


def test_a_zoom_on_one_chart_reaches_the_others_on_the_same_axis_and_a_reset_too(own_runs) -> None:
    lab, *_ = _lab(own_runs)
    _tabs(lab)["input-steering"].value = True
    lateral, steering = (lab._panes[_card_of(lab, p).id][0]
                         for p in ("output-e-lateral", "input-steering"))
    lateral.relayout_data = {"xaxis.range[0]": 0.2, "xaxis.range[1]": 0.8}
    assert steering.object["layout"]["xaxis"]["range"] == [0.2, 0.8]
    assert "range" not in lateral.object["layout"]["xaxis"], "the source is left alone"
    steering.relayout_data = {"xaxis.autorange": True}
    assert "range" not in lateral.object["layout"]["xaxis"]
    assert lateral.object["layout"]["xaxis"].get("autorange") is True


def test_a_bar_sets_the_centreline_on_its_children_and_the_form_is_one_set_of_tracks(own_runs) -> None:
    """The two shapes the page is laid out with, and why neither is the obvious one.

    A BAR is a row of text panes beside widgets. The centreline is set on the CHILDREN
    (`skin.BAR`, a stylesheet reaching the row's shadow root), never by `align="center"` on the
    row: Panel reads that as "align THIS ROW inside its container", and every child keeps its own
    `align`, default `start`, which beats any `align-items` the row carries. Measured on the live
    page 2026-10-06: without this a text pane sat 9.5-10px off the button beside it.

    The FORM is one grid per section wearing the SAME fixed tracks, so a box in RUN sits under a
    box in CONTROLLER -- tracks that size to their own content line up only within one grid, which
    left the boxes 196px apart across sections. The cells go in bare, three per field: a row of
    its own is a separate flex box and two of them share no geometry.
    """
    lab, *_ = _lab(own_runs)
    page = lab.page()

    bars = [r for r in page.select(pn.Row)
            if any(isinstance(o, pn.pane.HTML) for o in r)
            and any(isinstance(o, pn.widgets.Widget) for o in r)]
    assert len(bars) >= 4, f"the page's bars should be found, got {len(bars)}"
    for bar in bars:
        assert skin.BAR in (bar.stylesheets or []), \
            f"a bar whose children are left on their own align: {[type(o).__name__ for o in bar]}"
    assert not [r for r in page.select(pn.Row) if r.align == "center"], \
        "align='center' on a Row aligns the ROW in its container, not its children"

    grids = page.select(pn.GridBox)
    assert grids, "the form's sections are grids"
    per_group: dict[str, int] = {}
    for f in lab.form.fields:
        per_group[f["group"]] = per_group.get(f["group"], 0) + 1
    assert sorted(len(g.objects) for g in grids) == sorted(n * 3 for n in per_group.values()), \
        "three cells per field, and nothing else in the grid"
    for g in grids:
        assert g.ncols == 6 and skin.FORM_GRID in (g.stylesheets or []), \
            "every section wears the same six tracks, or the sections do not line up"


def test_the_seconds_the_page_is_on_outlive_the_next_render(own_runs) -> None:
    """A card is rebuilt from the record on every render and the figure layer draws it on all of
    its time, so the page has to put the zoom back or a tick would undo it."""
    lab, *_ = _lab(own_runs)
    pane = lambda p: lab._panes[_card_of(lab, p).id][0]
    pane("output-e-lateral").relayout_data = {"xaxis.range[0]": 0.2, "xaxis.range[1]": 0.8}
    _tick(lab, "first")                                   # a render: another run on every card
    assert pane("output-e-lateral").object["layout"]["xaxis"]["range"] == [0.2, 0.8]
    _tabs(lab)["input-steering"].value = True
    assert pane("input-steering").object["layout"]["xaxis"]["range"] == [0.2, 0.8], \
        "a chart opened while the others are zoomed is on the same seconds"
    pane("output-e-lateral").relayout_data = {"xaxis.autorange": True}
    _tick(lab, "first", on=False)
    assert "range" not in pane("output-e-lateral").object["layout"]["xaxis"], "a reset outlives it too"


def test_a_real_save_from_the_page_writes_the_still_beside_the_record(own_runs) -> None:
    """End to end: the page asks the real `uv run deepc draw`, and the files are where the
    terminal said."""
    root, out_dir = own_runs
    folder = Path(root) / out_dir / "closed_loop" / "first"
    shutil.rmtree(folder, ignore_errors=True)
    lab = Lab(root, out_dir, every=_Ticker(), toast=lambda _w: None, schedule=lambda work: work())
    lab.page()
    lab.run_list.shown = ["first"]
    _tabs(lab)["cost"].value = True
    _control(lab, "cost", "Table").value = True
    _save_button(lab, "cost").clicks += 1
    assert (folder / "cost.png").is_file() and (folder / "cost.png").stat().st_size > 1000
    assert (folder / "cost.png.json").is_file() and (folder / "cost.csv").is_file()
    assert f"{out_dir}/closed_loop/first/cost.png" in _console(lab) and "exit 0" in lab.last_line.object
    assert [f["name"] for f in lab.folders] == ["first"], "listed under the saved stills"


def test_the_page_renders_with_two_runs_and_every_kind_of_card(own_runs) -> None:
    """The page is BUILT here, not just driven: the figure dicts are validated by the Plotly
    pane when the document is rendered, after `page()` has returned."""
    lab, *_ = _lab(own_runs)
    _tick(lab, "first")
    _tabs(lab)["cost"].value = True
    _control(lab, "cost", "Split").value = True
    lab.open_number("solve_time_mean_s")
    lab.open_drawer("first")
    assert lab.page().get_root() is not None
    assert lab.split.left_px == 400
    lab.split.left_px = 520                          # what a drag sets; the page can set it too
    assert lab.split.left_px == 520


def test_a_record_that_cannot_be_read_is_not_shown_by_itself(own_runs) -> None:
    """The list gives a stale row no tick, so it must never be the run the screen shows."""
    root, out_dir = own_runs
    (Path(root) / out_dir / "closed_loop" / "zz-kept-setup.json").write_text('{"lambda_g": 2}')
    time.sleep(0.01)
    os.utime(Path(root) / out_dir / "closed_loop" / "zz-kept-setup.json", None)   # the newest file
    lab, *_ = _lab(own_runs)
    assert lab.shown == ["second"], "the newest READABLE run"
    assert _columns(lab) == ["second"]
    stale = next(r for r in lab.run_list.rows if r["name"] == "zz-kept-setup")
    assert "carries no" in stale["stale"]
    lab.run_list.shown = ["second", "zz-kept-setup"]   # a component cannot be trusted either
    assert lab.shown == ["second"]


def test_a_number_a_record_does_not_declare_well_is_listed_and_not_fatal(own_runs) -> None:
    root, out_dir = own_runs
    (Path(root) / out_dir / "closed_loop" / "zz-odd.json").write_text(json.dumps(
        {"component": "closed_loop", "panels": ["output-e-lateral"],
         "number_specs": {"rmse_position": 1}}))
    lab, *_ = _lab(own_runs)
    assert lab.page().get_root() is not None
    row = next(r for r in lab.runs() if r["name"] == "zz-odd")
    assert "'number_specs' header is not a list" in row["stale"] and row["number_specs"] == []


def test_switching_the_piece_forgets_what_was_ticked_and_open(own_runs) -> None:
    """Every run name belongs to the piece it came from."""
    lab, *_ = _lab(own_runs)
    _tick(lab, "first")
    _tabs(lab)["cost"].value = True
    lab._on_piece("dataset")                         # none of the closed loop's runs is the dataset's
    assert lab.shown == [] and lab.open_charts == []
    lab._on_piece("closed_loop")
    assert lab.shown == ["second"] and lab.open_charts == ["output-e-lateral"]
    assert not lab._loaded or all(p.endswith(".json") for p in lab._loaded)
