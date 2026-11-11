"""The open-loop piece: its timing (y_k = h(x_k)), its integration against the exact sampled
system, its excitation and actuator clip, its disturbance, its seed, its numbers (PE order, SNR),
what it refuses, and its record."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
from scipy.linalg import expm

from deepc.regularization import check_persistent_excitation
from harness import get_component
from harness.system import system_of

SYSTEM, OPEN = get_component("system"), get_component("open_loop")


@pytest.fixture(scope="module")
def systems(tmp_path_factory) -> dict[str, str]:
    """A saved default system run (the two masses) and a saved pendulum, by name: their paths."""
    out = tmp_path_factory.mktemp("systems")
    pendulum = SYSTEM.Config(f=["x2", "-sin(x1) - theta1*x2 + u1"], h=["x1"], theta=[0.2])
    return {"masses": str(SYSTEM.run(SYSTEM.Config()).save(out / "masses.json")),
            "pendulum": str(SYSTEM.run(pendulum).save(out / "pendulum.json"))}


def _run(systems, **settings):
    return OPEN.run(OPEN.Config(system_run=systems["masses"], **settings))


def test_the_output_at_sample_k_is_h_of_the_state_at_sample_k(systems) -> None:
    """Coulson et al. [1, eq. (1)]: y(t) = Cx(t) + Du(t), the output from the state at the same
    time. With no noise, y_k is h(x_k) exactly, from y_0 = h(x0)."""
    rec = _run(systems, sigma=0.0, T=50)
    plant = system_of(systems["masses"])
    assert np.array_equal(rec.x[0], rec.config.x0)
    assert all(np.array_equal(rec.y[k], plant.h(rec.x[k])) for k in range(50))
    assert np.array_equal(rec.t, np.arange(50) * 0.05)


def test_every_step_matches_the_exact_sampled_system_under_the_applied_input(systems) -> None:
    """The masses (θ = [k, k12, c, c12]) against the exact zero-order-hold sampling of the same
    linear system (C. F. Van Loan, IEEE TAC 23(3), 1978; not yet read), x_{k+1} = A_d x_k +
    B_d u_k with u_k the input the plant applied, at T_s = 0.05, no disturbance; the bound below
    is 1e-8."""
    rec = _run(systems, sigma=0.0, sigma_d=[0.0, 0.0], T=200)
    k, k12, c, c12 = SYSTEM.Config().theta
    A = np.array([[0, 1, 0, 0], [-(k + k12), -(c + c12), k12, c12],
                  [0, 0, 0, 1], [k12, c12, -(k + k12), -(c + c12)]])
    B = np.array([[0, 0], [1, 0], [0, 0], [0, 1]], float)
    M = expm(np.block([[A, B], [np.zeros((2, 4)), np.zeros((2, 2))]]) * 0.05)
    exact = rec.x[:-1] @ M[:4, :4].T + rec.u[:-1] @ M[:4, 4:].T
    assert np.abs(rec.x[1:] - exact).max() < 1e-8


def test_the_excitation_stays_in_its_range_and_the_bounds_clip_what_reaches_the_plant(systems) -> None:
    inside = _run(systems, T=300)
    assert inside.numbers["clipped"] == 0 and np.all(np.abs(inside.u) <= 3.0)
    wide = _run(systems, T=300, u_exc_min=[-8.0, -1.0], u_exc_max=[8.0, 1.0])
    assert np.all(np.abs(wide.u[:, 0]) <= 5.0) and np.any(np.abs(wide.u[:, 0]) == 5.0)
    assert wide.numbers["clipped"] == wide.constants["clipped_per_input"][0] > 0
    assert wide.constants["clipped_per_input"][1] == 0
    # the record holds what the plant applied, the clipped value, not the drawn one (D12)
    drawn = np.random.default_rng(42).uniform([-8.0, -1.0], [8.0, 1.0], size=(300, 2))
    assert np.array_equal(wide.u, np.clip(drawn, [-5.0, -5.0], [5.0, 5.0]))


def test_one_seed_one_record_and_another_seed_another(systems) -> None:
    a, b, c = _run(systems, T=100), _run(systems, T=100), _run(systems, T=100, seed=7)
    assert np.array_equal(a.u, b.u) and np.array_equal(a.y, b.y) and np.array_equal(a.x, b.x)
    assert not np.array_equal(a.u, c.u) and not np.array_equal(a.y, c.y)


@pytest.mark.parametrize("settings,words", [
    ({"x0": [0.0, 0.0]}, "x0 holds 2 values; the system has 4 states"),
    ({"u_min": [-5.0]}, "u_min holds 1 value; the system has 2 inputs"),
    ({"x0": [float("inf"), 0.0, 0.0, 0.0]}, "x0 = [inf, 0.0, 0.0, 0.0] is not finite"),
    ({"sigma_d": [0.1]}, "sigma_d holds 1 value; the system has 2 disturbances"),
    ({"Ts": float("inf")}, "Ts = [inf] is not finite"),
    ({"sigma": float("inf")}, "sigma = [inf] is not finite"),
    ({"sigma_d": [0.1, -0.1]}, "a standard deviation is not negative"),
    ({"u_min": [float("inf"), -5.0], "u_max": [float("inf"), 5.0]}, "u_min may be -inf"),
    ({"u_exc_max": [3.0, 3.0, 3.0]}, "u_exc_max holds 3 values"),
    ({"u_min": [6.0, -5.0]}, "u1: u_min 6.0 is above u_max 5.0"),
    ({"u_exc_min": [-3.0, 4.0]}, "u2: u_exc_min 4.0 is above u_exc_max 3.0"),
    ({"x0": [float("nan"), 0.0, 0.0, 0.0]}, "holds NaN"),
    ({"u_exc_min": [float("-inf"), -3.0]}, "an excitation is drawn from a finite range"),
])
def test_settings_that_do_not_fit_the_system_are_refused_in_words(systems, settings, words) -> None:
    with pytest.raises(ValueError, match=re.escape(words)):
        _run(systems, T=5, **settings)


def test_a_missing_or_stale_system_run_is_refused_with_the_command_that_writes_it(
        systems, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="deepc run system"):
        OPEN.run(OPEN.Config(system_run=str(tmp_path / "absent.json")))
    raw = json.loads(open(systems["masses"]).read())
    raw["constants"]["f_read"][0] = "x3"
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="does not read again to what it holds"):
        OPEN.run(OPEN.Config(system_run=str(stale)))


def test_a_nonlinear_system_runs_with_its_own_sizes_and_no_axis_carries_a_unit(systems, tmp_path) -> None:
    rec = OPEN.run(OPEN.Config(system_run=systems["pendulum"], x0=[1.0, 0.0], u_min=[-1.0],
                               u_max=[1.0], u_exc_min=[-0.5], u_exc_max=[0.5], sigma_d=[],
                               T=100))
    assert rec.u.shape == (100, 1) and rec.y.shape == (100, 1) and rec.x.shape == (100, 2)
    assert [s.name for s in rec.series()] == ["u1", "y1", "x1", "x2"]
    assert all(s.x_label == "t" and "[" not in s.y_label for s in rec.series())
    again = OPEN.Record.load(rec.save(tmp_path / "p.json"))
    for field in ("t", "u", "y", "x"):
        assert getattr(again, field).tobytes() == getattr(rec, field).tobytes()
    assert again.numbers == rec.numbers
    assert rec.constants["f_read"] == ["x2", "-theta1*x2 + u1 - sin(x1)"], \
        "the system as read, not only its path"
    assert rec.constants["theta"] == [0.2]


def test_a_run_with_no_actuator_limit_saves_and_loads_back(systems, tmp_path) -> None:
    """An infinite bound is a setting a run may hold (code review, 2026-11-11: it could not be
    saved, the settings skipping the record's non-finite tokens)."""
    rec = _run(systems, T=20, u_min=[float("-inf"), -5.0], u_max=[float("inf"), 5.0])
    again = OPEN.Record.load(rec.save(tmp_path / "open.json"))
    assert again.config.u_min == [float("-inf"), -5.0] and again.config.u_max == [float("inf"), 5.0]
    assert [r for r, _ in next(s for s in again.series() if s.name == "u1").refs] == []


def test_the_disturbance_moves_the_states_and_leaves_the_input_alone(systems) -> None:
    """The developer, 2026-11-11: "unmodelled disturbance is absent". d_k ~ N(0, σ_d²) is drawn
    after the excitation and the noise, held over each step where the rows name it."""
    quiet = _run(systems, T=100, sigma_d=[0.0, 0.0])
    pushed = _run(systems, T=100)
    assert np.array_equal(quiet.u, pushed.u) and np.array_equal(quiet.d, np.zeros((100, 2)))
    assert not np.array_equal(quiet.x[1:], pushed.x[1:]) and np.array_equal(quiet.x[0], pushed.x[0])
    assert pushed.d.shape == (100, 2) and 0.05 < pushed.d.std() < 0.2
    assert [s.name for s in pushed.series() if s.group == "Disturbances"] == ["d1", "d2"]


def test_the_snr_is_the_noise_free_output_variance_over_the_noise_variance(systems) -> None:
    rec = _run(systems, T=300)
    plant = system_of(systems["masses"])
    clean = np.array([plant.h(x) for x in rec.x])
    for j in range(4):
        assert rec.numbers[f"y{j + 1}_snr"] == pytest.approx(np.var(clean[:, j]) / 0.005 ** 2)
    silent = _run(systems, T=50, sigma=0.0)
    assert not any(k.endswith("_snr") for k in silent.numbers), "no noise: no ratio, not inf"


def test_the_pe_order_is_the_largest_depth_at_which_the_input_hankel_has_full_row_rank(
        systems) -> None:
    """Coulson et al. [1, Def. 4.4], by the controller's own check: an i.i.d. uniform input over
    T = 600 samples of m = 2 inputs reaches the largest depth the data allows, (T + 1)//(m + 1);
    an input constant on both channels is not persistently exciting of order 1."""
    rec = _run(systems)
    assert rec.numbers["pe_order"] == (600 + 1) // (2 + 1) == 200
    rank, needed = check_persistent_excitation(rec.u, 200, 2)
    assert rank == needed == 400
    constant = _run(systems, T=50, u_exc_min=[1.0, 2.0], u_exc_max=[1.0, 2.0])
    assert constant.numbers["pe_order"] == 0
