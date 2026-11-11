"""The two pieces drive today's code unchanged: each reproduces the safety net's frozen arrays bit
for bit, through its saved file (BESS's rule B4: a refactor proves itself by moving no number).
And each number on a record is a reduction of an array on it, a saved dataset run is checked
before it is used, and ``deepc run`` writes what it says.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import terminal
from harness import get_component
from tests.reference import cases

DATASET, LOOP = get_component("dataset"), get_component("closed_loop")
#: The closed-loop record's arrays and the frozen arrays they must equal.
_FROZEN = {"time": "times", "u": "u_history", "y": "y_history", "reference": "ref_path",
           "cost": "costs", "sigma_y_norm": "sigma_norms", "lambda_g": "lambda_g_history",
           "position": "pos_history", "reference_position": "ref_pos_history"}


@pytest.fixture(scope="module")
def datasets(tmp_path_factory) -> dict[str, str]:
    """A saved default dataset run per plant, as ``deepc run dataset --set plant=...`` writes it."""
    out = tmp_path_factory.mktemp("dataset")
    return {plant: str(DATASET.run(DATASET.Config(plant=plant)).save(out / f"{plant}.json"))
            for plant in ("bicycle", "coupled_masses")}


@pytest.mark.parametrize("plant", ["bicycle", "coupled_masses"])
def test_the_dataset_piece_reproduces_the_frozen_training_data(datasets, plant) -> None:
    rec = DATASET.Record.load(datasets[plant])
    frozen = np.load(cases.REFERENCE_DIR / f"{plant}-data.npz")
    assert rec.u.tobytes() == frozen["u_data"].tobytes()
    assert rec.y.tobytes() == frozen["y_data"].tobytes()


@pytest.mark.parametrize("case", ["bicycle-default", "coupled_masses-default-no-constraints"])
def test_the_closed_loop_piece_reproduces_the_frozen_loop(datasets, tmp_path, case) -> None:
    plant, scenario, duration, flags, overrides = cases.CLOSED_LOOPS[case]
    cfg = LOOP.Config(dataset_run=datasets[plant], scenario=scenario, sim_duration=duration,
                      **flags, **overrides)
    rec = LOOP.Record.load(LOOP.run(cfg).save(tmp_path / f"{case}.json"))
    frozen = np.load(cases.REFERENCE_DIR / f"{case}.npz")
    moved = [k for k, f in _FROZEN.items() if f in frozen.files
             and getattr(rec, k).tobytes() != frozen[f].astype(float).tobytes()]
    assert not moved and rec.statuses == frozen["statuses"].tolist(), f"{case} moved: {moved}"


def test_every_closed_loop_number_is_a_reduction_of_an_array_on_the_record(datasets) -> None:
    rec = LOOP.run(LOOP.Config(dataset_run=datasets["bicycle"], sim_duration=1.0))
    for j, c in enumerate(rec.outputs):
        assert rec.numbers[f"rmse_{c}"] == float(np.sqrt(np.mean(rec.y[:, j] ** 2)))
    counted = {k[len("solves_"):]: v for k, v in rec.numbers.items() if k.startswith("solves_")}
    assert counted == {s: float(n) for s, n in rec.status_counts().items()}
    assert sum(counted.values()) == len(rec.statuses) == len(rec.solve_time_s)
    assert rec.numbers["solve_time_max_s"] == float(rec.solve_time_s.max())
    assert rec.constants["plant_Ts_s"] == 0.1 and rec.constants["config_Ts_s"] == 0.1


def test_every_dataset_number_is_a_reduction_of_an_array_on_the_record(datasets) -> None:
    rec = DATASET.Record.load(datasets["coupled_masses"])
    for j, c in enumerate(rec.inputs):
        assert rec.numbers[f"u_rms_{c}"] == float(np.sqrt(np.mean(rec.u[:, j] ** 2)))
    assert rec.episode_starts == [0, 120, 240, 360, 480]


def test_a_saved_dataset_run_that_does_not_run_again_to_its_file_is_refused(datasets, tmp_path) -> None:
    raw = json.loads(open(datasets["bicycle"]).read())
    raw["u"][0][0] += 1e-12
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="does not run again"):
        LOOP.run(LOOP.Config(dataset_run=str(path), sim_duration=1.0))
    with pytest.raises(FileNotFoundError, match="deepc run dataset"):
        LOOP.run(LOOP.Config(dataset_run=str(tmp_path / "absent.json")))


def test_a_scenario_the_plant_does_not_offer_is_refused(datasets) -> None:
    with pytest.raises(ValueError, match="offers the scenarios"):
        LOOP.run(LOOP.Config(dataset_run=datasets["bicycle"], scenario="sinusoidal"))


def test_deepc_run_writes_one_record_per_swept_value(datasets, tmp_path, capsys) -> None:
    terminal.main(["run", "closed_loop", "--set", f"dataset_run={datasets['coupled_masses']}",
                   "--set", "sim_duration=0.5", "--sweep", "Tini=2,3", "--name", "s",
                   "--out-dir", str(tmp_path)])
    written = sorted(p.name for p in (tmp_path / "closed_loop").iterdir())
    assert written == ["s-Tini-2.json", "s-Tini-3.json"]
    assert LOOP.Record.load(tmp_path / "closed_loop" / "s-Tini-3.json").config.Tini == 3
    assert "solves_optimal" in capsys.readouterr().out
