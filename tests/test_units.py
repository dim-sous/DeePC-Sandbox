"""The units DeePC's plants and its older pieces declare (from ``tests/test_figure.py``, whose
checks of the figure layer moved to Benchlab's own suite with Benchlab's branch R1)."""

from __future__ import annotations

import pytest

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
