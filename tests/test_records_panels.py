"""What only the old ``run.py`` report drew, now declared by the records and drawn by the figure
layer: the bicycle's path in the plane, and the plant's input bounds on every input chart."""

from __future__ import annotations

import numpy as np
import pytest

from figure.chart import chart_of
from figure.render import table_of
from figure.twin import figure_of
from figure.chart import pool_refusal
from figure.render import render
from harness import get_component
from plants import PLANT_REGISTRY
from record import Series, stamp
from sim.bounds import input_bounds_of

DATASET, LOOP = get_component("dataset"), get_component("closed_loop")


@pytest.fixture(scope="module")
def loops(tmp_path_factory) -> dict:
    """A short closed loop per plant, each on its own saved default dataset run."""
    out = tmp_path_factory.mktemp("panels")
    runs = {}
    for plant in ("bicycle", "coupled_masses"):
        path = DATASET.run(DATASET.Config(plant=plant)).save(out / f"{plant}.json")
        scenario = "default" if plant == "bicycle" else "sinusoidal"
        runs[plant] = (DATASET.Record.load(path),
                       LOOP.run(LOOP.Config(dataset_run=str(path), sim_duration=0.5,
                                            scenario=scenario)))
    return runs


def test_the_bicycle_loop_declares_its_path_and_each_trace_keeps_its_own_x(loops, tmp_path) -> None:
    """The old report's "Trajectory (x-y)": y against x, the plant's positions (the start
    included) and the waypoints, each on its own abscissa, on equal scales."""
    _data, rec = loops["bicycle"]
    [path] = [s for s in rec.series() if s.name == "path"]
    assert path.equal_axes and [label for label, _ in path.traces] == ["Plant", "Reference"]
    panel = chart_of([("r", rec)], "path")
    plant, ref = panel.axes[0].lines
    assert np.array_equal(plant.x, rec.position[:, 0]) and np.array_equal(plant.y, rec.position[:, 1])
    assert np.array_equal(ref.x, rec.reference_position[:, 0])
    assert len(plant.x) == len(ref.x) + 1                    # the start, then one per step
    assert panel.equal_axes
    fig = figure_of(panel)
    assert fig["layout"]["yaxis"]["scaleanchor"] == "x"
    assert np.allclose(fig["data"][0]["x"], rec.position[:, 0])
    assert np.allclose(fig["data"][1]["x"], rec.reference_position[:, 0])
    assert fig["data"][0]["hovertemplate"] == "%{x:.6g}, %{y:.6g}"
    two = chart_of([("a", rec), ("b", rec)], "path")          # overlay: each run's own x
    assert len(two.axes[0].lines) == 2 and two.equal_axes     # identical traces drawn once each
    assert all(len(line.x) in (len(rec.position), len(rec.reference_position))
               for line in two.axes[0].lines)
    rows = chart_of([("a", rec), ("b", rec)], "path", runs="rows")
    assert [len(ax.lines) for ax in rows.axes] == [2, 2] and rows.equal_axes
    indexed = chart_of([("r", rec)], "path", x="index")       # metres against an index: unequal
    assert not indexed.equal_axes
    assert np.array_equal(indexed.axes[0].lines[0].x, np.arange(len(rec.position)))
    [png, *_] = render(panel, tmp_path, stamp(0.0))
    assert png.exists()
    assert {row[0] for row in table_of(panel).rows} == {"Plant", "Reference"}
    assert pool_refusal([("a", rec), ("b", rec)], "path", "mean") == "the chart declares no statistic"


def test_a_plant_without_positions_declares_no_path(loops) -> None:
    _data, rec = loops["coupled_masses"]
    assert "path" not in {s.name for s in rec.series()}


@pytest.mark.parametrize("plant", ["bicycle", "coupled_masses"])
def test_every_input_chart_carries_the_bounds_the_plant_clips_to(loops, plant) -> None:
    """The old report drew the bicycle's steering and acceleration limits. Every input chart,
    closed loop and dataset, now carries the plant's bounds as reference lines -- the bounds of
    ``get_constraints``, which the plant's own step clips to (checked here, input by input)."""
    model = PLANT_REGISTRY[plant]()
    bounds = input_bounds_of(model)
    for rec in loops[plant]:
        assert np.array_equal(rec.input_bounds, bounds)
        for j, (name, unit) in enumerate(zip(rec.inputs, rec.input_units)):
            [s] = [s for s in rec.series() if s.name == f"input-{name}".replace("_", "-")]
            assert dict(s.refs) == {f"Lower bound of {name} [{unit}]": bounds[j][0],
                                    f"Upper bound of {name} [{unit}]": bounds[j][1]}
    state = np.ones_like(model.state)           # moving, so every input changes the next state
    for j, (lo, hi) in enumerate(bounds):
        for beyond, at, inside in ((hi + 10.0, hi, hi - 0.01), (lo - 10.0, lo, lo + 0.01)):
            after = {}
            for name, value in (("beyond", beyond), ("at", at), ("inside", inside)):
                u = np.zeros(model.m)
                u[j] = value
                model.reset(state.copy())
                after[name] = model.step(u)
            assert np.array_equal(after["beyond"], after["at"]), f"{plant}: input {j} passes {at}"
            assert not np.array_equal(after["inside"], after["at"]), (
                f"{plant}: input {j} is clipped inside its bound {at}")


def test_an_infinite_bound_draws_no_line_and_survives_a_save(loops, tmp_path) -> None:
    """A plant may bound one side of an input only (``constraints_to_arrays`` gives the other
    side as an infinity): the record keeps it through a save, and its chart draws one line."""
    data, _rec = loops["bicycle"]
    rec = DATASET.Record.load(data.__class__.save(
        type(data)(**{**data.__dict__, "input_bounds": np.array([[-np.inf, 0.5], [-1.0, np.inf]])}),
        tmp_path / "half.json"))
    assert np.array_equal(rec.input_bounds, [[-np.inf, 0.5], [-1.0, np.inf]])
    refs = {s.name: dict(s.refs) for s in rec.series() if s.name.startswith("input-")}
    assert refs == {"input-steering": {"Upper bound of steering [rad]": 0.5},
                    "input-acceleration": {"Lower bound of acceleration [m/s²]": -1.0}}


def test_a_panel_whose_traces_each_have_an_abscissa_refuses_what_needs_one_shared() -> None:
    xs = (np.arange(3.0), np.arange(4.0))
    traces = (("a", np.zeros(3)), ("b", np.zeros(4)))
    Series("p", (), "x [m]", "y [m]", traces, trace_x=xs)
    with pytest.raises(ValueError, match="2 abscissae for 1 traces"):
        Series("p", (), "x [m]", "y [m]", traces[:1], trace_x=xs)
    with pytest.raises(ValueError, match="has 3 values against 4"):
        Series("p", (), "x [m]", "y [m]", traces, trace_x=xs[::-1])
    with pytest.raises(ValueError, match="needs one abscissa the traces share"):
        Series("p", (), "x [m]", "y [m]", traces, trace_x=xs, pooled=("mean",))
