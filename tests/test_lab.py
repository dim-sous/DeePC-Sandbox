"""The Lab's constraints, reads and launches (BESS's ``tests/test_lab.py``, adapted).

The Lab holds no logic of its own: it imports only what lets it read declarations and files and
launch the terminal, reaches no third-party package but Panel, and never calls anything that
could compute on what it reads. What it lists is read off the files; what it launches is the
line a person would type, every word checked against a declaration.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import shlex
import shutil
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import record
from harness import NAMES, get_component
from lab import jobs, state
from lab.app import Lab
from record.config import fields_of
from tests.lab_runs import make_runs

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "devkit" / "lab"
SOURCES = sorted(p for p in LAB.glob("*.py"))
#: Exactly what the Lab may take from the rest of the repo, by module and name.
_ALLOWED = {"harness": {"NAMES", "get_component"},
            # a record's header keys, and how every word the page shows begins (the developer's
            # rule, 2026-10-27)
            "record": {"read_header", "COMPONENT_KEY", "PANELS_KEY", "NUMBER_SPECS_KEY",
                       "PANEL_GROUPS_KEY", "HEADER_KEYS", "capital"},
            "record.config": {"fields_of"},
            # the one constant saying where records live, and the one list of the ways
            # `deepc draw` shapes a drawing, which a card's save is built from
            "terminal": {"DEFAULT_OUT_DIR", "DRAW_SHAPING"},
            # what a card draws is the figure layer's own: the panel built from what the records
            # declare and the card's controls hold, the refusals a control is greyed with, the
            # twin of it the page shows and the numbers on it; the colour a run wears is the
            # figure layer's slot, so its dot in the list matches its lines; the page's paper,
            # inks, rules and type are the figure layer's, so a chart sits on its card's paper;
            # the words of its sample-index axis are the figure layer's, which the X control offers
            "figure.chart": {"chart_of", "sweep_of", "pool_refusal", "distinct", "telling",
                             "INDEX_LABEL"},
            "figure.render": {"log_refusal", "table_of"},
            "figure.twin": {"figure_of"},
            "figure.style": {"series_color", "FONT", "MONO", "SURFACE", "INK", "INK_SECONDARY",
                             "GRID"}}
_THIRD_PARTY = {"panel", "param"}
_PROJECT = {p.name for r in ("src", "tools", "devkit") for p in (ROOT / r).iterdir()
            if (p / "__init__.py").exists()}
#: Call names that would make the Lab compute instead of read. Cheap to check and easy to
#: dodge by renaming, which is why the capability itself is held by the tripwire test below.
#: Loading a record through its class is READING (every rule of it the class's own), so
#: ``load`` is not here; building a trace is not reading, which the chart test holds.
_COMPUTES = {"run", "Config", "run_command", "draw_command"}


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory) -> tuple[str, str]:
    return make_runs(tmp_path_factory.mktemp("lab"))


@pytest.fixture
def own_runs(two_runs, tmp_path) -> tuple[str, str]:
    """A private copy of the two records. A test that draws writes sixteen files beside them,
    which the next test reads -- so each one that writes gets its own folder and the order these
    tests run in cannot matter."""
    root, out_dir = two_runs
    src, dst = Path(root) / out_dir / "closed_loop", tmp_path / out_dir / "closed_loop"
    dst.mkdir(parents=True)
    for name in ("first", "second"):
        shutil.copy(src / f"{name}.json", dst / f"{name}.json")
    return str(tmp_path), out_dir


def _top(name: str) -> str:
    return name.split(".")[0]


def _ended(job: jobs.Job, timeout: float = 20.0) -> None:
    """Wait for a launched job to end. One job runs per process, so a test that leaves one going
    would have the next test's launch refused."""
    deadline = time.time() + timeout
    while job.running and time.time() < deadline:
        time.sleep(0.05)
    assert not job.running, f"{job.line} still running after {timeout} s"


@pytest.mark.parametrize("path", SOURCES, ids=[p.name for p in SOURCES])
def test_the_lab_imports_only_its_allowlist_and_no_third_party_but_panel(path: Path) -> None:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = _top(alias.name)
                assert top in sys.stdlib_module_names or top in _THIRD_PARTY, \
                    f"{path.name} imports {alias.name}"
                assert top not in _PROJECT, f"{path.name}: `import {alias.name}` — only `from X import` of the allowlist"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top = _top(node.module)
            if top in sys.stdlib_module_names or top in _THIRD_PARTY:
                continue
            taken = {a.name for a in node.names}
            allowed = _ALLOWED.get(node.module, set())
            assert taken <= allowed, (f"{path.name} takes {sorted(taken - allowed)} from "
                                      f"{node.module}; allowed: {sorted(allowed)}")


def test_the_header_keys_the_lab_reads_are_the_records_own() -> None:
    """``state`` names the header keys one by one; if the record layer gains another, this fails
    instead of every rail row turning red with a false 'holds gone field'."""
    assert state._HEADER == record.HEADER_KEYS


def test_the_lab_never_calls_what_could_compute_on_what_it_reads() -> None:
    hits = []
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in _COMPUTES:
                hits.append(f"{path.name}:{node.lineno} {func.id}(")
            elif isinstance(func, ast.Attribute) and func.attr in _COMPUTES:
                owner = func.value.id if isinstance(func.value, ast.Name) else ""
                if owner not in {"json", "subprocess"}:
                    hits.append(f"{path.name}:{node.lineno} .{func.attr}(")
    assert not hits, hits


def _tripwire(real: types.ModuleType) -> types.ModuleType:
    """The piece with everything the Lab may READ -- its declarations, and its records through
    ``Record.load``, which is reading -- and a raise on everything that would COMPUTE: running
    it, building its Config."""

    def boom(*_a, **_k):
        raise AssertionError("the Lab computed instead of reading")

    class Config(real.Config):
        def __init__(self, *_a, **_k):
            boom()

    return types.SimpleNamespace(Record=real.Record, Config=Config, run=boom)


def test_driving_the_lab_never_runs_the_piece_nor_builds_its_config(monkeypatch, tmp_path, two_runs) -> None:
    """The page is driven through every handler a person reaches, against a piece that raises if
    it is run: the Lab may read a piece's declarations and must never exercise it."""
    real = get_component(NAMES[0])
    stand_in = _tripwire(real)
    for module in (state, jobs):
        monkeypatch.setattr(module, "get_component", lambda _n, s=stand_in: s)
    root, out_dir = two_runs
    launched: list[list[str]] = []

    def launch(argv, cwd=None):
        launched.append(list(argv))
        return jobs.Job(list(argv))

    lab = Lab(root, out_dir, launch=launch, every=lambda cb, ms: None, toast=lambda _w: None)
    lab.page()
    lab._on_type("lambda_g", "10")
    lab._on_pick("noise_adaptive", "true")
    lab._on_sweep(field="N", values="10,20")
    lab._on_name("probe")
    lab._use("second")
    lab._run()
    lab._poll()
    lab.reload()
    lab.run_list.shown = ["second", "first"]
    lab._on_chart("position-x", True)
    lab.open_number("rmse_position")
    for card in lab.cards:
        card.runs, card.table = ("mean" if card.kind == "chart" else "each"), True
    lab._draw_cards()
    lab._on_settings("all")
    assert launched and launched[0][:5] == ["uv", "run", "deepc", "run", NAMES[0]]
    assert not lab.banner.visible, lab.banner.object
    assert dataclasses.fields(stand_in.Record)      # the declarations WERE readable


def test_every_chart_on_the_page_is_the_figure_layers(monkeypatch, two_runs) -> None:
    """The page draws nothing of its own: with the figure layer's twin stubbed, every chart on
    the page is exactly what the stub returned, of exactly the panel the figure layer built for
    the shown runs' records in the order they were ticked."""
    import lab.app as app_module
    built: list[tuple[str, list[str]]] = []
    real_chart = app_module.chart_of

    def spy(pairs, name, **keywords):
        built.append((name, [run for run, _record in pairs]))
        return real_chart(pairs, name, **keywords)

    def stub(panel, legend=True):
        return {"data": [], "layout": {"height": 100, "meta": {"doc": f"stub {panel.name}",
                                                                "panel": panel.name}}}

    monkeypatch.setattr(app_module, "chart_of", spy)
    monkeypatch.setattr(app_module, "figure_of", stub)
    root, out_dir = two_runs
    lab = Lab(root, out_dir, every=lambda cb, ms: None, toast=lambda _w: None)
    lab.page()
    lab.run_list.shown = ["second", "first"]
    lab._on_chart("position-x", True)
    assert set(lab.open_charts) == {"output-e-lateral", "position-x"}
    cards = {c.id: c for c in lab.cards}
    for key, (pane, _label) in lab._panes.items():
        assert pane.object["layout"]["meta"]["doc"] == f"stub {cards[key].chart}"
    assert built[-1] == ("position-x", ["second", "first"]), "the shown runs, in the order ticked"


def test_the_pieces_are_the_registry_with_their_own_declarations() -> None:
    pieces = state.pieces()
    assert [p["name"] for p in pieces] == list(NAMES)
    for p in pieces:
        component = get_component(p["name"])
        assert p["fields"] == fields_of(component.Config)
        assert p["label"] == component.Record.label and p["produced_by"]
        assert all(f["group"] and f["label"] for f in p["fields"])


def test_an_empty_out_dir_is_an_empty_list_not_an_error(tmp_path: Path) -> None:
    assert state.records(str(tmp_path), "results") == []


def test_records_are_read_off_their_files_newest_first(two_runs) -> None:
    root, out_dir = two_runs
    rows = [r for r in state.records(root, out_dir) if r["piece"] == "closed_loop"]
    assert [r["name"] for r in rows] == ["second", "first"]
    for r in rows:
        assert set(r) == set(state.RECORD_KEYS)
        assert r["stale"] == "" and r["panels"]
        assert {n["name"] for n in r["number_specs"]} == set(r["numbers"])
        assert r["config"]["dataset_run"].endswith("dataset/default.json")
        assert r["provenance"]["git_sha"]
        json.dumps(r)                                    # plain data, nothing exotic
    assert rows[0]["config"]["lambda_g"] == 50 and rows[1]["config"]["lambda_g"] is None


def test_a_record_is_re_read_only_when_its_file_changes(two_runs) -> None:
    root, out_dir = two_runs
    before = next(r for r in state.records(root, out_dir) if r["name"] == "second")
    assert state.records(root, out_dir)[0] is before
    path = Path(before["path"])
    os.utime(path, (time.time() + 5, time.time() + 5))
    after = next(r for r in state.records(root, out_dir) if r["name"] == "second")
    assert after is not before and after["name"] == before["name"]


@pytest.mark.parametrize("body,words", [('{"a": 1}', "carries no 'component' header"),
                                         ("not json at all", "not JSON")])
def test_a_json_that_is_not_a_record_is_listed_with_the_refusal_as_its_words(
        tmp_path: Path, body: str, words: str) -> None:
    """A kept settings file or a stray sidecar must not take the page down: the row carries
    ``read_header``'s refusal and the rest of the folder still lists."""
    folder = tmp_path / "results" / "closed_loop"
    folder.mkdir(parents=True)
    (folder / "stray.json").write_text(body)
    rows = state.records(str(tmp_path), "results")
    assert len(rows) == 1 and rows[0]["name"] == "stray"
    assert words in rows[0]["stale"] and rows[0]["config"] == {} and rows[0]["panels"] == []


def test_a_record_the_class_has_moved_past_is_listed_and_marked(two_runs, tmp_path: Path) -> None:
    root, out_dir = two_runs
    raw = json.loads((Path(root) / out_dir / "closed_loop" / "first.json").read_text())
    raw.pop("sigma_y_norm")
    raw["mood"] = [1, 2]
    folder = tmp_path / "results" / "closed_loop"
    folder.mkdir(parents=True)
    (folder / "old.json").write_text(json.dumps(raw))
    [row] = state.records(str(tmp_path), "results")
    assert "predates the field sigma_y_norm" in row["stale"] and "holds gone field mood" in row["stale"]


def test_a_run_becomes_the_command_a_person_would_type() -> None:
    assert jobs.run_argv("closed_loop", ["lambda_g=10", "N=10"], "probe") == [
        "uv", "run", "deepc", "run", "closed_loop", "--set", "lambda_g=10", "--set", "N=10",
        "--name", "probe"]
    assert jobs.run_argv("closed_loop", [], "p", "elsewhere", ["lambda_g=1,5"]) == [
        "uv", "run", "deepc", "run", "closed_loop", "--sweep", "lambda_g=1,5", "--name", "p",
        "--out-dir", "elsewhere"]


def test_the_echoed_line_reads_back_into_the_launched_argv() -> None:
    argv = jobs.run_argv("closed_loop", ["lambda_g=10"], "a.b-c_1")
    assert shlex.split(jobs.line_of(argv)) == argv


@pytest.mark.parametrize("piece,overrides,name,words", [
    ("bogus", [], "x", "unknown piece 'bogus'"),
    ("closed_loop", ["nope=1"], "x", "'nope' is not a setting of closed_loop"),
    ("closed_loop", ["lambda_g"], "x", "is not SETTING=VALUE"),
    ("closed_loop", ["lambda_g="], "x", "is not SETTING=VALUE"),
    ("closed_loop", [], "", "cannot name a record"),
    ("closed_loop", [], "--out-dir", "cannot name a record"),
    ("closed_loop", [], "../x", "cannot name a record"),
    ("closed_loop", [], "a b", "cannot name a record"),
])
def test_only_a_declared_setting_and_a_file_stem_can_become_a_run(piece, overrides, name, words) -> None:
    with pytest.raises(ValueError, match=words):
        jobs.run_argv(piece, overrides, name)


def test_a_job_whose_output_cannot_be_read_still_records_how_it_ended() -> None:
    """Whatever the child prints, the exit code is recorded: a job that latched ``running`` for
    ever would grey Run, never list what it wrote, and refuse every later run."""
    job = jobs.launch([sys.executable, "-c",
                       "import sys; sys.stdout.buffer.write(b'ok\\n\\xff\\n')"])
    _ended(job)
    assert job.returncode == 0 and job.output[0] == "ok\n"


def test_one_run_per_process_is_refused_in_words() -> None:
    """Each browser session builds its own page, so without this two tabs would run two
    ``deepc run`` at once -- racing the venv and, under one name, one record file."""
    first = jobs.launch([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(ValueError, match="a command is already in progress"):
            jobs.launch([sys.executable, "-c", "pass"])
    finally:
        first.stop()
        _ended(first)
    after = jobs.launch([sys.executable, "-c", "pass"])      # the next one is free to go
    _ended(after)
    assert after.returncode == 0


def test_a_job_streams_its_output_and_can_be_stopped() -> None:
    job = jobs.launch([sys.executable, "-u", "-c",
                       "import time; print('started'); time.sleep(30); print('never')"])
    deadline = time.time() + 10
    while not job.output and time.time() < deadline:
        time.sleep(0.05)
    assert job.output == ["started\n"] and job.running and not job.stopped
    assert job.stop() is True and job.stopped
    _ended(job)
    assert job.returncode != 0 and "never" not in "".join(job.output)
    assert job.stop() is False


def test_a_job_that_was_never_started_cannot_be_stopped() -> None:
    job = jobs.Job(["uv", "run", "deepc"])
    assert not job.running and job.stop() is False and job.line == "uv run deepc"


def test_a_draw_names_the_record_files_and_every_flag_the_terminal_declares(two_runs) -> None:
    root, out_dir = two_runs
    first, second = (os.path.join(out_dir, "closed_loop", f"{n}.json") for n in ("first", "second"))
    # the paths stay relative, as a person would type them from the root; `root` is where the
    # records are looked for, which is not the directory the server runs in
    assert jobs.draw_argv("closed_loop", ["first"], out_dir, root=root) == [
        "uv", "run", "deepc", "draw", first]
    argv = jobs.draw_argv("closed_loop", ["first", "second"], out_dir, root=root, panels=["position-x"],
                          table=True, runs="one", colour_by="quantity", points=False,
                          log_y=None, log_x=False, x="index", x_range=(60.0, 300.0))
    assert argv == ["uv", "run", "deepc", "draw", first, second, "--panel", "position-x", "--runs", "one",
                    "--colour-by", "quantity", "--no-points", "--x", "index",
                    "--x-range=60.0,300.0", "--table"], "a None and a False flag pass nothing"
    below = jobs.draw_argv("closed_loop", ["first"], out_dir, root=root, x_range=(-5.0, 10.0))
    from terminal import build_parser
    assert build_parser().parse_args(below[3:]).x_range == (-5.0, 10.0), \
        "a span from below zero reaches the terminal as one (code review, 2026-10-27)"
    assert jobs.draw_argv("closed_loop", ["first"], out_dir, root=root, number="rmse_position",
                          vs="lambda_g", by=()) == [
        "uv", "run", "deepc", "draw", first, "--number", "rmse_position", "--vs", "lambda_g",
        "--by"]
    with pytest.raises(ValueError, match="no record at"):
        jobs.draw_argv("closed_loop", ["first"], out_dir)            # not from here
    with pytest.raises(ValueError, match="no record at"):
        jobs.draw_argv("closed_loop", ["absent"], out_dir, root=root)
    with pytest.raises(ValueError, match="cannot name a record"):
        jobs.draw_argv("closed_loop", ["../x"], out_dir, root=root)
    with pytest.raises(ValueError, match="not ways `deepc draw` shapes"):
        jobs.draw_argv("closed_loop", ["first"], out_dir, root=root, colour="red")


def test_a_card_and_its_still_are_drawn_from_the_same_keywords(own_runs) -> None:
    """What the page hands the figure layer for a card is what its save hands ``deepc draw``:
    every keyword a card's controls produce is a flag the terminal declares."""
    from lab import shaping
    from terminal import DRAW_SHAPING
    declared = {k for k, *_ in DRAW_SHAPING}
    for kind in ("chart", "numbers"):
        card = shaping.Card(1, kind, chart="position-x", number="rmse_position", x="lambda_g")
        ctx = shaping.Context(kind, runs=2, axes={"λ_g": "lambda_g"}, telling={"N": "N"})
        assert set(shaping.draw_keywords(card, ctx)) <= declared, kind


def test_delete_takes_only_what_it_names_and_refuses_anything_else(own_runs) -> None:
    root, out_dir = own_runs
    folder = Path(root) / out_dir / "closed_loop"
    (folder / "first").mkdir()
    (folder / "first" / "position-x.png").write_bytes(b"png")
    (folder / "first" / "position-x.png.json").write_text("{}")
    (folder / "kept").mkdir()
    (folder / "kept" / "notes.txt").write_text("mine")
    with pytest.raises(ValueError, match="which no draw writes"):
        jobs.delete("closed_loop", folders=["kept"], root=root, out_dir=out_dir)
    with pytest.raises(ValueError, match="no record 'absent'"):
        jobs.delete("closed_loop", records=["absent"], root=root, out_dir=out_dir)
    with pytest.raises(ValueError, match="cannot name a record"):
        jobs.delete("closed_loop", records=["../second"], root=root, out_dir=out_dir)
    assert (folder / "kept" / "notes.txt").exists() and (folder / "second.json").exists()
    with pytest.raises(ValueError, match="not a folder a draw writes"):
        jobs.delete("closed_loop", folders=["../first"], root=root, out_dir=out_dir)
    (folder / "first+second").mkdir()
    (folder / "first+second" / "cost.png").write_bytes(b"png")
    gone = jobs.delete("closed_loop", records=["first"], folders=["first", "first+second"],
                       root=root, out_dir=out_dir)
    assert gone == [os.path.join(out_dir, "closed_loop", "first.json"),
                    os.path.join(out_dir, "closed_loop", "first"),
                    os.path.join(out_dir, "closed_loop", "first+second")], \
        "the folder a draw of two runs writes (code review, 2026-10-27)"
    assert not (folder / "first.json").exists() and not (folder / "first").exists()
    assert (folder / "second.json").exists()


def test_a_draw_of_many_runs_writes_a_folder_a_filesystem_and_delete_both_take() -> None:
    from terminal import folder_of
    assert folder_of(["a", "b"]) == "a+b"
    names = [f"default-lambda_g-{20 + i}-run-{i}" for i in range(20)]
    many = folder_of(names)
    assert len(many.encode()) <= 255 and many.startswith(names[0] + "+19-more-")
    assert many != folder_of(names[:-1] + ["other"]), "two sets of runs, two folders"
    assert jobs._FOLDER.match(many)


def test_the_folders_of_stills_are_listed_with_what_they_hold(own_runs) -> None:
    root, out_dir = own_runs
    folder = Path(root) / out_dir / "closed_loop" / "first+second"
    folder.mkdir()
    for name in ("position-x.png", "position-x.png.json", "position-x.csv", "position-x.csv.json",
                 "cost.png"):
        (folder / name).write_text("x")
    (Path(root) / out_dir / "closed_loop" / "empty").mkdir()
    [listed] = state.folders(root, out_dir, "closed_loop")
    assert listed["name"] == "first+second" and listed["pictures"] == ["cost.png", "position-x.png"]
    assert listed["tables"] == 1


def test_one_command_at_a_time_in_either_direction(own_runs) -> None:
    """Two ``uv run`` at once race the venv, so the second is refused in words, not queued --
    a draw beside a run, a run beside a draw, or two draws."""
    root, out_dir = own_runs
    running = jobs.launch([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(ValueError, match="a command is in progress"):
            jobs.capture(jobs.draw_argv("closed_loop", ["first"], out_dir, root=root), cwd=root)
    finally:
        running.stop()
        _ended(running)
    code, output = jobs.capture([sys.executable, "-c", "print('after')"])
    assert code == 0 and "after" in output

    # the other direction, and two draws: `capture` claims the slot as well as checking it
    held = threading.Thread(target=lambda: jobs.capture(
        [sys.executable, "-c", "import time; time.sleep(1.5)"]), daemon=True)
    held.start()
    time.sleep(0.4)
    try:
        with pytest.raises(ValueError, match="a command is in progress"):
            jobs.capture([sys.executable, "-c", "pass"])
        with pytest.raises(ValueError, match="a command is already in progress"):
            jobs.launch([sys.executable, "-c", "pass"])
    finally:
        held.join(timeout=20)
    assert jobs.capture([sys.executable, "-c", "pass"])[0] == 0


def test_a_shown_record_is_loaded_through_its_own_class_in_its_own_words(own_runs) -> None:
    """Loading is reading: the class decodes the file and refuses one it cannot, in words."""
    root, out_dir = own_runs
    run = next(r for r in state.records(root, out_dir) if r["name"] == "first")
    record = state.load(run)
    assert type(record) is get_component("closed_loop").Record
    assert [s.name for s in record.series()] == run["panels"]
    assert record.constants["plant_Ts_s"] > 0, "what the compare table's constants read"
    broken = Path(root) / out_dir / "closed_loop" / "broken.json"
    broken.write_text(Path(run["path"]).read_text().replace('"time": [', '"tim": [', 1))
    with pytest.raises(ValueError, match="does not match the current Record"):
        state.load(dict(run, path=str(broken)))


def test_a_path_outside_the_out_dir_is_not_under_it(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    assert state.under(str(tmp_path / "results" / "x.json"), str(tmp_path / "results"))
    assert not state.under(str(tmp_path / "results" / ".." / "secret"), str(tmp_path / "results"))
    assert not state.under(str(tmp_path / "results-sibling" / "x"), str(tmp_path / "results"))


def test_wrote_keeps_only_the_printed_lines_that_are_files_under_the_out_dir(own_runs) -> None:
    root, out_dir = own_runs
    rel = os.path.join(out_dir, "closed_loop", "first.json")
    lines = [rel + "\n", "  rmse_position = 0.09\n", "../etc/hostname\n", "\n"]
    assert state.wrote(lines, root, out_dir) == [os.path.join(root, rel)]
