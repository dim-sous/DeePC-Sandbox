"""The closed-loop piece: its plant is the open-loop run's plant, sampled by the same functions
(one quantity, one formula: D4, D10, D12); its warm-up, its own seed, its reference; a solve with
no solution stops it; what it refuses; its record and numbers. And the fundamental lemma on the
Hankel blocks the controller is built from."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
from scipy.linalg import expm

from deepc.controller import DeePCController
from deepc.hankel import build_data_matrices
from harness import get_component
from harness.system import system_of
from system import parse_signal

SYSTEM, OPEN, LOOP = get_component("system"), get_component("open_loop"), get_component("closed_loop")
#: Short loops: 10 solves after the T_ini = 4 warm-up at T_s = 0.05.
SHORT = 0.5


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, str]:
    """A saved default system run (the two masses), an open-loop run on it at T = 300 and its
    other settings at their defaults, and one with no noise, no disturbance and no actuator
    limit: their paths."""
    out = tmp_path_factory.mktemp("loop")
    system = str(SYSTEM.run(SYSTEM.Config()).save(out / "masses.json"))
    data = OPEN.run(OPEN.Config(system_run=system, T=300)).save(out / "open.json")
    clean = OPEN.run(OPEN.Config(system_run=system, T=300, sigma=0.0, sigma_d=[0.0, 0.0],
                                 u_min=[-np.inf] * 2, u_max=[np.inf] * 2)).save(out / "clean.json")
    return {"system": system, "open": str(data), "clean": str(clean)}


def _loop(runs, data="open", **settings):
    return LOOP.run(LOOP.Config(open_loop_run=runs[data], **{"duration": SHORT, **settings}))


@pytest.fixture(scope="module")
def loop(runs):
    return _loop(runs)


def test_the_loop_measures_as_the_open_loop_does_with_its_own_seed(runs, loop) -> None:
    """y_k = h(x_k) + n_k, the open-loop run's formula (``system.measure``) and σ; the noise is
    drawn from the loop's seed, not the data's (the developer, 2026-11-17)."""
    plant = system_of(runs["system"])
    data = OPEN.Record.load(runs["open"])
    K = len(loop.t)
    assert all(np.array_equal(loop.y_true[k], plant.h(loop.x[k])) for k in range(K))
    noise = data.config.sigma * np.random.default_rng(loop.config.seed).standard_normal((K, 4))
    assert np.allclose(loop.y - loop.y_true, noise, rtol=0, atol=1e-15)
    assert loop.config.seed != data.config.seed == loop.constants["data_seed"]
    assert np.array_equal(loop.x[0], data.x[0]), "x_0 left at none is the open-loop run's"
    assert np.array_equal(loop.t, np.arange(K) * data.config.Ts)


def test_the_plant_in_the_loop_is_the_open_loop_runs(runs, loop) -> None:
    """The loop's T_s, bounds, σ, σ_d and tolerances are the open-loop run's, recorded; and on
    the noise-free run each step is the exact zero-order-hold sampling of the masses (as
    ``tests/test_open_loop.py`` holds for the open loop; Van Loan 1978, not yet read), bound
    1e-8."""
    oc = OPEN.Record.load(runs["open"]).config
    held = loop.constants
    assert (held["Ts"], held["u_min"], held["u_max"], held["sigma"], held["sigma_d"],
            held["rtol"], held["atol"]) == (oc.Ts, oc.u_min, oc.u_max, oc.sigma, oc.sigma_d,
                                           oc.rtol, oc.atol)
    rec = _loop(runs, data="clean")
    k, k12, c, c12 = SYSTEM.Config().theta
    A = np.array([[0, 1, 0, 0], [-(k + k12), -(c + c12), k12, c12],
                  [0, 0, 0, 1], [k12, c12, -(k + k12), -(c + c12)]])
    B = np.array([[0, 0], [1, 0], [0, 0], [0, 1]], float)
    M = expm(np.block([[A, B], [np.zeros((2, 4)), np.zeros((2, 2))]]) * 0.05)
    exact = rec.x[:-1] @ M[:4, :4].T + rec.u[:-1] @ M[:4, 4:].T
    assert np.abs(rec.x[1:] - exact).max() < 1e-8


def test_the_warm_up_is_zero_and_every_later_sample_is_solved(loop) -> None:
    Tini = loop.config.Tini
    assert not loop.u[:Tini].any()
    assert len(loop.statuses) == len(loop.solvers) == len(loop.solve_time) == len(loop.t) - Tini
    assert loop.constants["steps"] == int(SHORT / 0.05) == len(loop.statuses)


def test_a_duration_that_is_a_multiple_of_the_step_gives_that_many_solves(runs) -> None:
    """0.35 / 0.05 reads 6.999999999999999 in floats; the old count int(duration / T_s) gave 6
    solves (code review, 2026-11-17). A duration between two multiples gives the solves it holds."""
    assert _loop(runs, duration=0.35).constants["steps"] == 7
    assert _loop(runs, duration=0.37).constants["steps"] == 7


def test_one_seed_one_record_and_another_seed_another(runs, loop) -> None:
    again, other = _loop(runs), _loop(runs, seed=7)
    assert np.array_equal(again.y, loop.y) and np.array_equal(again.u, loop.u)
    assert not np.array_equal(other.y, loop.y)


def test_the_reference_is_the_typed_rows_at_the_sample_times(loop) -> None:
    _read, signal = parse_signal(loop.config.r)
    assert np.array_equal(loop.r, signal(loop.t))
    chart = next(s for s in loop.series() if s.name == "y1")
    assert [name for name, _ in chart.traces] == ["y1", "r1"]


def test_a_solve_with_no_solution_stops_the_run_naming_the_sample(runs, monkeypatch) -> None:
    """The developer, 2026-11-17: a solve with no answer stops the run; no stand-in input is
    applied (the controller's own zeros, flagged, are not used)."""
    def no_answer(self, *_a, **_k):
        return np.zeros(2), {"no_solution": True, "status": "infeasible", "solver": "OSQP"}
    monkeypatch.setattr(DeePCController, "solve", no_answer)
    with pytest.raises(RuntimeError, match=re.escape("the solve at sample 4 (t = 0.2) left no "
                                                     "solution: status 'infeasible' from OSQP")):
        _loop(runs)


@pytest.mark.parametrize("settings,words", [
    ({"Q": [1.0, 1.0]}, "Q holds 2 values; the system has 4 outputs"),
    ({"R": [0.1]}, "R holds 1 value; the system has 2 inputs"),
    ({"Q": [1.0, -1.0, 1.0, 1.0]}, "a weight may not be negative"),
    ({"x0": [0.0, 0.0]}, "x0 holds 2 values; the system has 4 states"),
    ({"r": ["0", "0"]}, "r holds 2 rows; the system has 4 outputs"),
    ({"r": ["x1", "0", "0", "0"]}, "r1 = 'x1' names ['x1']"),
    ({"duration": 0.01}, "shorter than one step T_s = 0.05"),
    ({"lambda_g": float("inf")}, "lambda_g = inf is not finite"),
])
def test_settings_that_do_not_fit_are_refused_in_words(runs, settings, words) -> None:
    with pytest.raises(ValueError, match=re.escape(words)):
        _loop(runs, **settings)


def test_a_missing_or_stale_open_loop_run_is_refused_with_the_command_that_writes_it(
        runs, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="deepc run open_loop"):
        LOOP.run(LOOP.Config(open_loop_run=str(tmp_path / "absent.json")))
    raw = json.loads(open(runs["open"]).read())
    raw["y"][3][0] += 1.0
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="does not run again to what it holds"):
        LOOP.run(LOOP.Config(open_loop_run=str(stale), duration=SHORT))


def test_a_system_run_written_again_after_a_loop_ran_is_refused_at_the_next(tmp_path) -> None:
    """One process (the Lab's runner, a sweep): a loop runs on an open-loop run, then its system
    file is written again with another θ; the next loop on the same open-loop file is refused,
    not built on data from one plant while it simulates another (code review, 2026-11-17)."""
    system = str(SYSTEM.run(SYSTEM.Config()).save(tmp_path / "masses.json"))
    data = str(OPEN.run(OPEN.Config(system_run=system, T=100)).save(tmp_path / "open.json"))
    LOOP.run(LOOP.Config(open_loop_run=data, duration=0.1))
    SYSTEM.run(SYSTEM.Config(theta=[3.0, 1.0, 0.3, 0.1])).save(tmp_path / "masses.json")
    with pytest.raises(ValueError, match="does not run again to what it holds"):
        LOOP.run(LOOP.Config(open_loop_run=data, duration=0.1))


def test_every_number_is_a_reduction_of_an_array_on_the_record(loop) -> None:
    """The RMSE is taken on h(x_k) - r_k, the plant's own output (CLAUDE.md: the plant's state
    generates the scores), over every sample, warm-up included."""
    n = loop.numbers
    for i in range(4):
        assert n[f"y{i + 1}_rmse"] == float(np.sqrt(np.mean((loop.y_true[:, i] - loop.r[:, i]) ** 2)))
    assert sum(v for k, v in n.items() if k.startswith("solves_") and not k.startswith("solves_by_")) \
        == sum(v for k, v in n.items() if k.startswith("solves_by_")) == len(loop.statuses)
    assert (n["solve_time_mean"], n["solve_time_std"], n["solve_time_max"]) == (
        float(loop.solve_time.mean()), float(loop.solve_time.std()), float(loop.solve_time.max()))
    assert (n["sigma_y_norm_mean"], n["sigma_y_norm_max"]) == (
        float(loop.sigma_y_norm.mean()), float(loop.sigma_y_norm.max()))
    assert n["clipped"] == sum(loop.constants["clipped_per_input"])
    assert {s.name for s in loop.number_specs()} == set(n)


def test_the_record_saves_and_loads_back_and_no_axis_carries_a_unit(loop, tmp_path) -> None:
    again = LOOP.Record.load(loop.save(tmp_path / "loop.json"))
    for field in ("t", "u", "y", "y_true", "r", "x", "d", "cost", "sigma_y_norm", "solve_time"):
        assert getattr(again, field).tobytes() == getattr(loop, field).tobytes(), field
    assert again.statuses == loop.statuses and again.numbers == loop.numbers
    names = [s.name for s in loop.series()]
    assert names == ["y1", "y2", "y3", "y4", "u1", "u2", "x1", "x2", "x3", "x4", "d1", "d2",
                     "solve-time", "cost", "sigma-y-norm"]
    assert all(s.x_label == "t" and "[" not in s.y_label for s in loop.series())
    assert loop.constants["deepc_config"]["Tini"] == 4 and "sim_duration" not in \
        loop.constants["deepc_config"], "only the settings the controller reads are recorded"


def test_the_hankel_blocks_of_noise_free_data_predict_the_systems_own_outputs(runs) -> None:
    """The fundamental lemma (Coulson et al. [1, Lemma 4.2]; van Waarde et al. [6, Theorem 1]):
    on noise-free data of a linear system whose input is persistently exciting of order
    T_ini + N + n, any g with U_p g = u_ini, Y_p g = y_ini, U_f g = u_f gives Y_f g = the system's
    own future outputs. Checked on the masses' noise-free open-loop run against a fresh
    trajectory from another x_0 and another input (seed 9); bound 1e-6, against the integrator's
    rtol 1e-9."""
    data = OPEN.Record.load(runs["clean"])
    Tini, N = 4, 20
    Up, Yp, Uf, Yf = build_data_matrices(data.u, data.y, Tini, N)
    fresh = OPEN.run(OPEN.Config(system_run=runs["system"], T=Tini + N, sigma=0.0,
                                 sigma_d=[0.0, 0.0], x0=[-0.3, 0.2, 0.4, -0.1], seed=9))
    lhs = np.vstack([Up, Yp, Uf])
    rhs = np.concatenate([fresh.u[:Tini].ravel(), fresh.y[:Tini].ravel(), fresh.u[Tini:].ravel()])
    g = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    assert np.abs(lhs @ g - rhs).max() < 1e-9
    assert np.abs(Yf @ g - fresh.y[Tini:].ravel()).max() < 1e-6
