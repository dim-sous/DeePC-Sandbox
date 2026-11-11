"""What only the old ``run.py`` report drew, now declared by the records and drawn by the figure
layer: the plant's input bounds on every input chart."""

from __future__ import annotations

import numpy as np
import pytest

from harness import get_component
from plants import PLANT_REGISTRY
from sim.bounds import input_bounds_of

DATASET, LOOP = get_component("dataset"), get_component("closed_loop")


@pytest.fixture(scope="module")
def loops(tmp_path_factory) -> dict:
    """A short closed loop per plant, each on its own saved default dataset run."""
    out = tmp_path_factory.mktemp("panels")
    runs = {}
    for plant in PLANT_REGISTRY:
        path = DATASET.run(DATASET.Config(plant=plant)).save(out / f"{plant}.json")
        runs[plant] = (DATASET.Record.load(path),
                       LOOP.run(LOOP.Config(dataset_run=str(path), sim_duration=0.5,
                                            scenario="sinusoidal")))
    return runs


@pytest.mark.parametrize("plant", list(PLANT_REGISTRY))
def test_every_input_chart_carries_the_bounds_the_plant_clips_to(loops, plant) -> None:
    """The old report drew the input limits of the plant it drew. Every input chart,
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
    data, _rec = loops["coupled_masses"]
    rec = DATASET.Record.load(data.__class__.save(
        type(data)(**{**data.__dict__, "input_bounds": np.array([[-np.inf, 0.5], [-1.0, np.inf]])}),
        tmp_path / "half.json"))
    assert np.array_equal(rec.input_bounds, [[-np.inf, 0.5], [-1.0, np.inf]])
    refs = {s.name: dict(s.refs) for s in rec.series() if s.name.startswith("input-")}
    assert refs == {"input-F1": {"Upper bound of F1 [N]": 0.5},
                    "input-F2": {"Lower bound of F2 [N]": -1.0}}
