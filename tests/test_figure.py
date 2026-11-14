"""The figure layer draws what records declare, and figures describe: axis labels naming the
quantity (no units on the new pieces; the old ones keep theirs until rewritten), a legend only when
there is more than one label, no title (CLAUDE.md, "In figures"; skill ``figure-conventions``)."""

from __future__ import annotations

import dataclasses
import json
import re

import numpy as np
import pytest

import benchlab.figure.render as render_module
from benchlab import cli
from benchlab.figure import draw
from benchlab.figure.chart import chart_of, sweep_of
from benchlab.figure.render import has_legend
from harness import get_component
from plants import PLANT_REGISTRY

DATASET, LOOP = get_component("dataset"), get_component("closed_loop")


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, object]:
    """A saved dataset run per plant and a 0.5 s closed loop on each, saved and loaded back. Every
    loop in this module runs the ``sinusoidal`` scenario: the coupled masses' ``default`` steps hold
    zero for their first 5 s, so a shorter loop on them charts round-off around zero."""
    out = tmp_path_factory.mktemp("runs")
    got = {}
    for plant in PLANT_REGISTRY:
        data = DATASET.run(DATASET.Config(plant=plant)).save(out / f"data-{plant}.json")
        got[f"data-{plant}"] = DATASET.Record.load(data)
        loop = LOOP.run(LOOP.Config(dataset_run=str(data), sim_duration=0.5, scenario="sinusoidal"))
        got[f"loop-{plant}"] = LOOP.Record.load(loop.save(out / f"loop-{plant}.json"))
    return got


@pytest.mark.parametrize("name", list(PLANT_REGISTRY))
def test_every_plant_declares_one_unit_per_channel(name: str) -> None:
    plant = PLANT_REGISTRY[name]()
    assert len(plant.input_units) == plant.m and len(plant.output_units) == plant.p
    assert all(plant.input_units) and all(plant.output_units)
    assert all(plant.metric_units.values())


def test_every_axis_of_the_old_pieces_still_carries_a_unit(runs) -> None:
    """The rule was "axis labels with units"; the developer replaced it on 2026-11-11 with "no
    units anywhere" (``tests/test_system.py`` holds it for the system piece). The dataset and
    closed-loop pieces on the plant classes keep their units until the rewrite replaces them
    (``docs/project/design.md`` §6)."""
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
    runs["data-coupled_masses"].save(data)
    other = LOOP.run(LOOP.Config(dataset_run=str(data), sim_duration=0.5, scenario="sinusoidal",
                                 lambda_g=2.0))
    panel = chart_of([("a", runs["loop-coupled_masses"]), ("b", other)], "input-F1")
    assert [ln.label for ln in panel.axes[0].lines] == ["a", "b"]   # a run alone names its trace
    assert len({ln.color for ln in panel.axes[0].lines}) == 2
    assert [r.label for r in panel.axes[0].refs] == ["Lower bound of F1 [N]",
                                                     "Upper bound of F1 [N]"]   # each bound once
    same = chart_of([("a", runs["loop-coupled_masses"]), ("b", runs["loop-coupled_masses"])],
                    "output-e-x1")
    assert [ln.label for ln in same.axes[0].lines] == ["a, b — e_x1 (identical)"]


def test_a_panel_no_run_declares_is_refused(runs, tmp_path) -> None:
    with pytest.raises(ValueError, match="no run declares"):
        draw([("a", runs["loop-coupled_masses"])], tmp_path, panels=["no-such-panel"])


def test_a_run_without_the_panel_or_of_another_piece_is_refused_not_left_out(runs) -> None:
    loop = runs["loop-coupled_masses"]

    class WithoutCost(type(loop)):
        """A run of the same piece that declares no ``cost`` panel: with one plant registered,
        no real run lacks a panel another declares."""

        def series(self):
            return tuple(s for s in super().series() if s.name != "cost")

    fewer = WithoutCost(**{f.name: getattr(loop, f.name) for f in dataclasses.fields(loop)})
    with pytest.raises(ValueError, match="declare no panel"):
        chart_of([("all", loop), ("fewer", fewer)], "cost")
    with pytest.raises(ValueError, match="different pieces"):
        chart_of([("data", runs["data-coupled_masses"]), ("loop", loop)], "input-F1")
    with pytest.raises(ValueError, match="different pieces"):
        sweep_of([("data", runs["data-coupled_masses"]), ("loop", loop)], "rmse_e_x1",
                 vs="seed")


def test_a_sweep_puts_one_point_per_run_at_its_settings_value(runs, tmp_path) -> None:
    data = tmp_path / "data.json"
    runs["data-coupled_masses"].save(data)
    pairs = [(f"lg-{v}", LOOP.run(LOOP.Config(dataset_run=str(data), sim_duration=0.5,
                                              scenario="sinusoidal", lambda_g=v)))
             for v in (2.0, 5.0)]
    panel = sweep_of(pairs, "rmse_e_x1", vs="lambda_g")
    (line,) = panel.axes[0].lines                         # one colour: only lambda_g differs
    assert list(line.drawn_x()) == [2.0, 5.0] and line.points
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
    cli.main(["draw", str(path), "--panel", "output-e-x1", "--out-dir", str(tmp_path / "out")])
    printed = capsys.readouterr().out.split()
    assert printed == [str(tmp_path / "out" / "output-e-x1.png")]
    sidecar = json.loads((tmp_path / "out" / "output-e-x1.png.json").read_text())
    assert sidecar["runs"] == ["loop"] and "git_sha" in sidecar
    assert np.isfinite(runs["loop-coupled_masses"].numbers["rmse_e_x1"])


@pytest.mark.parametrize("flags,words", [
    (["--vs", "lambda_g"], "--vs shapes a --number draw"),
    (["--vs", ""], "--vs shapes a --number draw"),
    (["--number", "rmse_e_x1"], "--number needs --vs"),
    (["--number", "rmse_e_x1", "--vs", "N", "--panel", "cost"], "--number draws one summary number"),
    (["--runs", "each"], "runs is one of"),
    (["--number", "rmse_e_x1", "--vs", "N", "--runs", "rows"], "runs is one of"),
])
def test_deepc_draw_refuses_a_flag_its_draw_would_not_use(runs, tmp_path, flags, words) -> None:
    """``--vs`` and ``--runs`` shape one kind of draw each: given to the other, or left out
    where it is needed, the draw is refused in words, not drawn without it."""
    path = tmp_path / "loop.json"
    runs["loop-coupled_masses"].save(path)
    with pytest.raises(SystemExit, match=re.escape(words)):
        cli.main(["draw", str(path), *flags, "--out-dir", str(tmp_path / "out")])


def test_a_number_against_a_list_setting_places_each_list_by_name(runs, tmp_path) -> None:
    """Q and R are lists: a sweep over them puts each list on the bottom axis by name, as a
    choice is put, rather than raising (a list cannot key a dict; code review, 2026-11-11)."""
    data = tmp_path / "data.json"
    runs["data-coupled_masses"].save(data)
    other = LOOP.run(LOOP.Config(dataset_run=str(data), sim_duration=0.5, scenario="sinusoidal",
                                 Q_diag=[10.0, 1.0, 10.0, 1.0]))
    loop = runs["loop-coupled_masses"]
    panel = sweep_of([("a", loop), ("b", other)], "rmse_e_x1", vs="Q_diag", by=())
    assert [words for _, words in panel.x_ticks] == ["None", "[10.0, 1.0, 10.0, 1.0]"]
    [line] = panel.axes[0].lines
    assert list(line.x) == [0.0, 1.0]
    assert list(line.y) == [loop.numbers["rmse_e_x1"], other.numbers["rmse_e_x1"]]


def test_a_log_axis_the_record_declares_reaches_both_drawings(tmp_path) -> None:
    """A Bode plot reads ω and |G| in decades: the record declares the axes logarithmic, and the
    still and the Lab's twin draw them so; a card offers no log control (the developer's cut)."""
    from benchlab.figure.render import render
    from benchlab.figure.twin import figure_of
    from benchlab.record import stamp
    system = get_component("system")
    rec = system.run(system.Config())
    magnitude = chart_of([("a", rec)], "bode-magnitude")
    assert magnitude.x_log and magnitude.y_log
    twin = figure_of(magnitude)
    assert twin["layout"]["xaxis"]["type"] == "log" and twin["layout"]["yaxis"]["type"] == "log"
    phase = figure_of(chart_of([("a", rec)], "bode-phase"))
    assert phase["layout"]["xaxis"]["type"] == "log" and "type" not in phase["layout"]["yaxis"]
    step = figure_of(chart_of([("a", rec)], "step"))
    assert "type" not in step["layout"]["xaxis"]
    [png, *_] = render(magnitude, tmp_path, stamp(0.0))
    assert png.exists()
