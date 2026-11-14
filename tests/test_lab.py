"""The Lab's reads and runs, driven through DeePC's own pieces (BESS's ``tests/test_lab.py``,
adapted). The Lab is Benchlab's: the half of this file that read the Lab's own files (its import
allowlist, and that it never calls what could compute) moved there with it, and so did the tests
of how a run is started and followed (Benchlab's ``tests/test_runner.py``).

The Lab holds no logic of its own. What it lists is read off the files; what it queues is checked,
word by word, against a declaration, and each run is its own process.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import threading
import time
import types
from pathlib import Path

import pytest

from harness import NAMES, get_component
from benchlab.lab import jobs, state
from benchlab.lab.app import Lab
from benchlab.record.config import fields_of
from tests.lab_runs import make_runs


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


class _Unstarted:
    """A run's process, stood in for: it never runs a piece and ends only when stopped."""

    returncode = None

    def __init__(self) -> None:
        self._ended = threading.Event()

    def communicate(self) -> tuple[str, str]:
        self._ended.wait()
        return "", ""

    def terminate(self) -> None:
        self.returncode = -15
        self._ended.set()


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
    started: list[jobs.Run] = []
    queue = jobs.Queue(at_once=1, start=lambda run: started.append(run) or _Unstarted())
    lab = Lab(root, out_dir, queue=queue, every=lambda cb, ms: None, toast=lambda _w: None)
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
    lab._on_chart("input-F1", True)
    lab.open_number("rmse_position_combined")
    for card in lab.cards:
        card.runs, card.table = ("mean" if card.kind == "chart" else "each"), True
    lab._draw_cards()
    lab._on_settings("all")
    # Use loaded the second run's settings, and with them took back the name and sweep typed
    [run] = queue.runs()
    assert run.piece == NAMES[0] and run.line.startswith(f"uv run deepc run {NAMES[0]} ")
    assert started == [run], "queued and started as its own process, here a stand-in"
    assert not lab.banner.visible, lab.banner.object
    queue.stop_all()
    assert dataclasses.fields(stand_in.Record)      # the declarations WERE readable


def test_every_chart_on_the_page_is_the_figure_layers(monkeypatch, two_runs) -> None:
    """The page draws nothing of its own: with the figure layer's twin stubbed, every chart on
    the page is exactly what the stub returned, of exactly the panel the figure layer built for
    the shown runs' records in the order they were ticked."""
    import benchlab.lab.app as app_module
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
    lab._on_chart("input-F1", True)
    assert set(lab.open_charts) == {"output-e-x1", "input-F1"}
    import panel as pn
    drawn = {}
    for card in lab.cards_box.objects:                 # each card's chart, by its title's panel
        chart = re.search(r"data-panel='([^']+)'", card.objects[0].objects[0].object).group(1)
        drawn[chart] = next(o for o in card.objects if isinstance(o, pn.pane.Plotly)).object
    assert set(drawn) == {"output-e-x1", "input-F1"}
    for chart, fig in drawn.items():
        assert fig["layout"]["meta"]["doc"] == f"stub {chart}"
    assert built[-1] == ("input-F1", ["second", "first"]), "the shown runs, in the order ticked"


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
    assert rows[0]["config"]["lambda_g"] == 2 and rows[1]["config"]["lambda_g"] is None


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


def test_delete_takes_only_what_it_names_and_refuses_anything_else(own_runs) -> None:
    root, out_dir = own_runs
    folder = Path(root) / out_dir / "closed_loop"
    with pytest.raises(ValueError, match="no record 'absent'"):
        jobs.delete("closed_loop", records=["absent"], root=root, out_dir=out_dir)
    with pytest.raises(ValueError, match="cannot name a record"):
        jobs.delete("closed_loop", records=["../second"], root=root, out_dir=out_dir)
    assert (folder / "second.json").exists()
    gone = jobs.delete("closed_loop", records=["first"], root=root, out_dir=out_dir)
    assert gone == [os.path.join(out_dir, "closed_loop", "first.json")]
    assert not (folder / "first.json").exists() and (folder / "second.json").exists()


def test_a_draw_of_many_runs_writes_a_folder_a_filesystem_takes() -> None:
    from benchlab.cli import folder_of
    assert folder_of(["a", "b"]) == "a+b"
    names = [f"default-lambda_g-{20 + i}-run-{i}" for i in range(20)]
    many = folder_of(names)
    assert len(many.encode()) <= 255 and many.startswith(names[0] + "+19-more-")
    assert many != folder_of(names[:-1] + ["other"]), "two sets of runs, two folders"


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
