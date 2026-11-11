"""The figure layer draws what records declare, and figures describe: axis labels with units (one
unit per axis), a legend only when there is more than one label, no title (CLAUDE.md, "In
figures"; skill ``figure-conventions``; D7 for a bound's unit)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import figure.render as render_module
import terminal
from figure import draw
from figure.chart import chart_of, sweep_of
from figure.render import has_legend
from harness import get_component
from plants import PLANT_REGISTRY

DATASET, LOOP = get_component("dataset"), get_component("closed_loop")


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, object]:
    """A saved dataset run per plant and a 1 s closed loop on each, saved and loaded back."""
    out = tmp_path_factory.mktemp("runs")
    got = {}
    for plant in ("bicycle", "coupled_masses"):
        data = DATASET.run(DATASET.Config(plant=plant)).save(out / f"data-{plant}.json")
        got[f"data-{plant}"] = DATASET.Record.load(data)
        loop = LOOP.run(LOOP.Config(dataset_run=str(data), sim_duration=1.0))
        got[f"loop-{plant}"] = LOOP.Record.load(loop.save(out / f"loop-{plant}.json"))
    return got


@pytest.mark.parametrize("name", list(PLANT_REGISTRY))
def test_every_plant_declares_one_unit_per_channel(name: str) -> None:
    plant = PLANT_REGISTRY[name]()
    assert len(plant.input_units) == plant.m and len(plant.output_units) == plant.p
    assert all(plant.input_units) and all(plant.output_units)
    assert all(plant.metric_units.values())


def test_every_axis_a_record_declares_carries_a_unit(runs) -> None:
    for key, rec in runs.items():
        for s in rec.series():
            for label in (s.x_label, s.y_label):
                assert label.endswith("]") and " [" in label, f"{key} {s.name}: {label!r}"


def test_every_declared_panel_draws_with_no_title_and_a_legend_only_for_several_labels(
        runs, tmp_path, monkeypatch) -> None:
    seen = []
    keep = render_module.save_figure

    def spy(fig, out_dir, stem, facts, prov):
        ax = fig.axes[0]
        seen.append((stem, ax.get_title(), ax.get_legend() is not None, list(fig.texts)))
        return keep(fig, out_dir, stem, facts, prov)

    monkeypatch.setattr(render_module, "save_figure", spy)
    for key, rec in runs.items():
        paths = draw([(key, rec)], tmp_path / key)
        assert len(paths) == len(rec.series())
        for p in paths:
            assert p.exists() and (p.parent / f"{p.name}.json").exists()
        for s in rec.series():
            panel = chart_of([(key, rec)], s.name)
            stem, title, legend, suptitle = next(x for x in seen if x[0] == s.name)
            assert title == "" and suptitle == []
            labels = {t for t, _ in s.traces} | {r for r, _ in s.refs} | {b[0] for b in s.bands}
            assert legend == has_legend(panel) == (len(labels) > 1)
        seen.clear()


def test_an_overlay_names_each_run_and_draws_what_both_hold_identically_once(runs, tmp_path) -> None:
    data = tmp_path / "data.json"
    runs["data-bicycle"].save(data)
    other = LOOP.run(LOOP.Config(dataset_run=str(data), sim_duration=1.0, lambda_g=50.0))
    panel = chart_of([("a", runs["loop-bicycle"]), ("b", other)], "position-x")
    labels = [ln.label for ln in panel.axes[0].lines]
    assert labels == ["a", "a, b — Reference (identical)", "b"]   # a run alone names its own trace
    assert len({ln.color for ln in panel.axes[0].lines}) == 3


def test_a_panel_no_run_declares_is_refused(runs, tmp_path) -> None:
    with pytest.raises(ValueError, match="no run declares"):
        draw([("a", runs["loop-bicycle"])], tmp_path, panels=["no-such-panel"])


def test_a_run_without_the_panel_or_of_another_piece_is_refused_not_left_out(runs) -> None:
    with pytest.raises(ValueError, match="declare no panel"):
        chart_of([("bike", runs["loop-bicycle"]), ("masses", runs["loop-coupled_masses"])],
                 "position-x")
    with pytest.raises(ValueError, match="different pieces"):
        chart_of([("data", runs["data-bicycle"]), ("loop", runs["loop-bicycle"])], "input-steering")
    with pytest.raises(ValueError, match="different pieces"):
        sweep_of([("data", runs["data-bicycle"]), ("loop", runs["loop-bicycle"])], "rmse_x",
                 vs="seed")


def test_a_sweep_puts_one_point_per_run_at_its_settings_value(runs, tmp_path) -> None:
    data = tmp_path / "data.json"
    runs["data-coupled_masses"].save(data)
    pairs = [(f"lg-{v}", LOOP.run(LOOP.Config(dataset_run=str(data), sim_duration=0.5, lambda_g=v)))
             for v in (2.0, 20.0)]
    panel = sweep_of(pairs, "rmse_e_x1", vs="lambda_g")
    (line,) = panel.axes[0].lines                         # one colour: only lambda_g differs
    assert list(line.drawn_x()) == [2.0, 20.0] and line.points
    assert line.y == [rec.numbers["rmse_e_x1"] for _, rec in pairs]
    spec = next(n for n in pairs[0][1].number_specs() if n.name == "rmse_e_x1")
    assert panel.axes[0].y_label == f"{spec.label} [m]"
    with pytest.raises(ValueError, match="not a setting"):
        sweep_of(pairs, "rmse_e_x1", vs="lambda_gg")
    with pytest.raises(ValueError, match="declare no number"):
        sweep_of(pairs, "rmse_nothing", vs="lambda_g")


def test_deepc_draw_writes_what_it_prints(runs, tmp_path, capsys) -> None:
    path = tmp_path / "loop.json"
    runs["loop-coupled_masses"].save(path)
    terminal.main(["draw", str(path), "--panel", "output-e-x1", "--out-dir", str(tmp_path / "out")])
    printed = capsys.readouterr().out.split()
    assert printed == [str(tmp_path / "out" / "output-e-x1.png")]
    sidecar = json.loads((tmp_path / "out" / "output-e-x1.png.json").read_text())
    assert sidecar["runs"] == ["loop"] and "git_sha" in sidecar
    assert np.isfinite(runs["loop-coupled_masses"].numbers["rmse_e_x1"])


def test_a_number_against_a_list_setting_places_each_list_by_name(runs, tmp_path) -> None:
    """Q and R are lists: a sweep over them puts each list on the bottom axis by name, as a
    choice is put, rather than raising (a list cannot key a dict; code review, 2026-11-11)."""
    data = tmp_path / "data.json"
    runs["data-bicycle"].save(data)
    other = LOOP.run(LOOP.Config(dataset_run=str(data), sim_duration=1.0, Q_diag=[10.0, 10.0, 1.0]))
    panel = sweep_of([("a", runs["loop-bicycle"]), ("b", other)], "rmse_position", vs="Q_diag",
                    by=())
    assert [words for _, words in panel.x_ticks] == ["None", "[10.0, 10.0, 1.0]"]
    [line] = panel.axes[0].lines
    assert list(line.x) == [0.0, 1.0]
    assert list(line.y) == [runs["loop-bicycle"].numbers["rmse_position"], other.numbers["rmse_position"]]
