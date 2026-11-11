"""The typed system: what the parser reads and refuses, the integrator against known answers, and
the system piece's record."""

from __future__ import annotations

import re

import numpy as np
import pytest
from scipy.linalg import expm

from harness import get_component
from record.config import with_overrides
from system import ATOL, RTOL, hold, parse

SYSTEM = get_component("system")


def _masses():
    cfg = SYSTEM.Config()
    return parse(cfg.f, cfg.h, cfg.theta)


def test_the_default_system_reads_as_typed_with_theta_kept_as_symbols() -> None:
    plant = _masses()
    assert (plant.n, plant.m, plant.p, plant.q, plant.r) == (4, 2, 4, 4, 2)
    assert plant.theta == (2.0, 1.0, 0.3, 0.1)
    assert "theta1" in plant.f_read[1] and plant.h_read == ("x1", "x2", "x3", "x4")


@pytest.mark.parametrize("f,h,theta,words", [
    (["__import__('os')"], ["x1"], [], "only numbers"),
    (["x1.real + u1"], ["x1"], [], "'.' outside a number"),
    (["lambda + u1"], ["x1"], [], "names ['lambda']"),
    (["eval(x1) + u1"], ["x1"], [], "names ['eval']"),
    (["x1; u1"], ["x1"], [], "only numbers"),
    (["2x1 + u1"], ["x1"], [], "is not an expression"),
    (["x2 + u1"], ["x1"], [], "x2 is named, but there are 1 state row(s)"),
    (["-x1 + u1"], ["x1 + u1"], [], "an output row names an input"),
    (["-x1"], ["x1"], [], "no state row names an input"),
    (["-x1 + u2"], ["x1"], [], "u1 acts in no state row as read"),
    (["-x1 + 0*u1"], ["x1"], [], "u1 acts in no state row as read (-x1)"),
    (["-x1 + u1", "  "], ["x1"], [], "x2' is empty"),
    ([], ["x1"], [], "no state row"),
    (["-x1 + u1"], [], [], "no output row"),
    (["(x1 + u1"], ["x1"], [], "is not an expression (TokenError"),
    (["1/0 + u1"], ["x1"], [], "is not an expression (ZeroDivisionError"),
    (["1e999*x1 + u1"], ["x1"], [], "holds the number 1e999, which is not finite"),
    (["-theta2*x1 + u1"], ["x1"], [1.0], "theta2 is named, but θ holds 1 value(s)"),
    (["-theta1*x1 + u1"], ["x1"], [1.0, 2.0], "θ2 = 2.0 acts in no row as read"),
    (["0*theta1*x1 + u1"], ["x1"], [1.0], "θ1 = 1.0 acts in no row as read"),
    (["x1 + sqrt(-1)*u1"], ["x1"], [], "which is not real"),
    (["()", "u1"], ["x1"], [], "is not an expression (it reads as ()"),
    (["9^9^9*x1 + u1"], ["x1"], [], "holds a constant too large for a float"),
    (["-theta1*x1 + u1"], ["x1"], [float("nan")], "θ1 = nan is not finite"),
    (["-x1 + u1"], ["x1 + d1"], [], "an output row names a disturbance"),
    (["-x1 + u1 + 0*d1"], ["x1"], [], "d1 acts in no state row as read"),
    (["-x1 + u1 + d2"], ["x1"], [], "d1 acts in no state row as read"),
])
def test_what_the_parser_cannot_read_is_refused_in_words(f, h, theta, words) -> None:
    """``parse_expr`` evaluates Python, so anything outside the expression grammar is refused
    before SymPy sees it: names, attribute access, separators, other characters. Numbers are read
    as floats, because SymPy computes an integer power exactly and ``9^9^9`` would not finish; as
    a float it reads at once, too large, and is refused."""
    with pytest.raises(ValueError, match=re.escape(words)):
        parse(f, h, theta)


def test_the_integrator_matches_the_exact_sampled_linear_system() -> None:
    """The masses as typed, 600 steps of T_s = 0.05, inputs uniform on [-5, 5] held per step
    (seed 42), from x0 = [0.5, 0, -0.5, 0], against the exact zero-order-hold sampling of the
    same linear system: the matrix exponential of the augmented matrix [[A, B], [0, 0]] T_s
    (C. F. Van Loan, IEEE TAC 23(3), 1978; not yet read). Measured 2026-11-11 at these
    tolerances: largest state error 3.4e-10; the bound below is 1e-8."""
    plant, Ts = _masses(), 0.05
    k, k12, c, c12 = plant.theta
    A = np.array([[0, 1, 0, 0], [-(k + k12), -(c + c12), k12, c12],
                  [0, 0, 0, 1], [k12, c12, -(k + k12), -(c + c12)]])
    B = np.array([[0, 0], [1, 0], [0, 0], [0, 1]], float)
    M = expm(np.block([[A, B], [np.zeros((2, 4)), np.zeros((2, 2))]]) * Ts)
    Ad, Bd = M[:4, :4], M[:4, 4:]
    U = np.random.default_rng(42).uniform(-5, 5, size=(600, 2))
    x = exact = np.array([0.5, 0.0, -0.5, 0.0])
    worst = 0.0
    no_d = np.zeros(plant.r)                              # the nominal plant: no disturbance
    for u in U:
        x = hold(lambda x_, u_: plant.f(x_, u_, no_d), x, u, (0.0, Ts))[1][-1]
        exact = Ad @ exact + Bd @ u
        worst = max(worst, float(np.abs(x - exact).max()))
    assert (RTOL, ATOL) == (1e-9, 1e-12) and worst < 1e-8, worst


def test_the_integrator_matches_closed_forms() -> None:
    decay = parse(["-theta1*x1 + u1"], ["x1"], [1.0])
    t, x = hold(lambda x_, u_: decay.f(x_, u_, np.zeros(0)), np.array([2.0]), np.zeros(1),
                (0.0, 3.0))
    assert np.allclose(x[:, 0], 2.0 * np.exp(-t), rtol=1e-8, atol=1e-11)
    ramp = parse(["u1 + d1"], ["x1"], [])                 # x1' = u1 + d1, both held
    assert hold(lambda x_, u_: ramp.f(x_, u_, np.array([0.25])), np.array([1.0]),
                np.array([0.5]), (0.0, 2.0))[1][-1] == pytest.approx([2.5], abs=1e-11)


def test_the_system_run_records_its_rows_as_read_and_its_sizes(tmp_path) -> None:
    rec = SYSTEM.run(SYSTEM.Config())
    assert {k: rec.numbers[k] for k in "nmpqr"} == {"n": 4, "m": 2, "p": 4, "q": 4, "r": 2}
    assert rec.constants["h_read"] == ["x1", "x2", "x3", "x4"]
    again = SYSTEM.Record.load(rec.save(tmp_path / "s.json"))
    assert again.config.f == rec.config.f and again.numbers == rec.numbers


def test_a_list_of_rows_round_trips_through_the_command_line_words() -> None:
    cfg = with_overrides(SYSTEM.Config(), [("f", '["x2", "-sin(x1) - theta1*x2 + u1"]'),
                                           ("h", '["x1"]'), ("theta", "[0.2]")])
    assert cfg.f == ["x2", "-sin(x1) - theta1*x2 + u1"] and cfg.theta == [0.2]
    cfg = with_overrides(cfg, [("x_star", "[0, 0]"), ("u_star", "[0]")])
    assert {k: SYSTEM.run(cfg).numbers[k] for k in "nmpqr"} == {"n": 2, "m": 1, "p": 1, "q": 1,
                                                                  "r": 0}
    with pytest.raises(ValueError, match="is not a JSON list of texts"):
        with_overrides(SYSTEM.Config(), [("f", "x2, -x1 + u1")])


# --- the analysis at a requested point (S1) ----------------------------------------------------

PENDULUM = {"f": ["x2", "-sin(x1) - theta1*x2 + u1"], "h": ["x1"], "theta": [0.2], "u_star": [0.0]}


def test_the_linearisation_is_exact_at_the_requested_point() -> None:
    """The masses are linear: A, B, C are θ's matrices wherever they are linearised. The
    pendulum hangs at (0, 0) and stands at (π, 0): A's lower-left entry is -cos(x1*)."""
    from system import linearize
    plant = _masses()
    k, k12, c, c12 = plant.theta
    A, B, C, residual = linearize(plant, [0.3, -1.0, 2.0, 0.5], [1.0, -2.0])
    assert np.array_equal(A, [[0, 1, 0, 0], [-(k + k12), -(c + c12), k12, c12],
                              [0, 0, 0, 1], [k12, c12, -(k + k12), -(c + c12)]])
    assert np.array_equal(B, [[0, 0], [1, 0], [0, 0], [0, 1]]) and np.array_equal(C, np.eye(4))
    assert residual > 0, "not an equilibrium: the masses are not at rest there"
    pendulum = parse(PENDULUM["f"], PENDULUM["h"], PENDULUM["theta"])
    down, _, _, at_rest = linearize(pendulum, [0.0, 0.0], [0.0])
    up, *_ = linearize(pendulum, [np.pi, 0.0], [0.0])
    assert np.array_equal(down, [[0, 1], [-1, -0.2]]) and at_rest == 0.0
    assert up[1, 0] == pytest.approx(1.0) and up[1, 1] == -0.2


def test_the_bode_and_step_responses_are_the_pairs_own(tmp_path) -> None:
    """The frequency response against G(jω) = C (jωI - A)^-1 B computed here, at the recorded ω;
    the step response settles at the DC gain, -C A^-1 B, for the stable masses (pair u1 -> y1)."""
    rec = SYSTEM.run(SYSTEM.Config())
    A, B, C = (np.array(rec.constants[k]) for k in ("A", "B", "C"))
    G = np.array([(C[[0], :] @ np.linalg.solve(1j * w * np.eye(4) - A, B[:, [0]]))[0, 0]
                  for w in rec.omega])
    assert np.allclose(rec.magnitude, np.abs(G), rtol=1e-9)
    assert np.allclose(np.mod(rec.phase_deg - np.degrees(np.angle(G)) + 180, 360) - 180, 0,
                       atol=1e-6)
    dc = float(-(C[[0], :] @ np.linalg.solve(A, B[:, [0]]))[0, 0])
    assert rec.numbers["dc_gain"] == pytest.approx(dc) == pytest.approx(0.375)
    assert rec.step[-1] == pytest.approx(dc, rel=2e-2)
    again = SYSTEM.Record.load(rec.save(tmp_path / "analysed.json"))
    assert again.magnitude.tobytes() == rec.magnitude.tobytes()


def test_the_pole_numbers_say_whether_the_point_is_stable() -> None:
    hanging = SYSTEM.run(SYSTEM.Config(**PENDULUM, x_star=[0.0, 0.0]))
    standing = SYSTEM.run(SYSTEM.Config(**PENDULUM, x_star=[float(np.pi), 0.0]))
    assert hanging.numbers["unstable_poles"] == 0 and hanging.numbers["max_pole_re"] < 0
    assert hanging.numbers["min_damping"] == pytest.approx(0.1)        # s² + 0.2 s + 1
    assert standing.numbers["unstable_poles"] == 1
    assert standing.numbers["max_pole_re"] == pytest.approx(-0.1 + np.sqrt(1.01))   # s² + 0.2 s - 1
    assert hanging.numbers["residual"] == 0.0 and standing.numbers["residual"] < 1e-15
    assert standing.numbers["dc_gain"] == pytest.approx(-1.0)


def test_a_number_with_no_finite_answer_is_left_out_says_why_and_the_run_saves(tmp_path) -> None:
    """A pole at the origin has no damping ratio and an integrator no DC gain (code review,
    2026-11-11: the NaN damping went into the numbers and the run could not be saved)."""
    integrator = SYSTEM.run(SYSTEM.Config(f=["x2", "u1"], h=["x1"], theta=[]))
    assert "dc_gain" not in integrator.numbers and "min_damping" not in integrator.numbers
    assert integrator.constants["left_out"]["dc_gain"].startswith("python-control gives")
    assert "undefined" in integrator.constants["left_out"]["min_damping"]
    assert "dc_gain" not in {n.name for n in integrator.number_specs()}
    again = SYSTEM.Record.load(integrator.save(tmp_path / "integrator.json"))
    assert again.numbers == integrator.numbers


def test_a_pair_the_input_never_reaches_is_refused_in_words() -> None:
    with pytest.raises(ValueError, match="u2 does not reach y1 at"):
        SYSTEM.run(SYSTEM.Config(f=["-x1 + u1", "-x2 + u2"], h=["x1", "x2"], theta=[], input=2))


def test_reading_a_saved_system_does_not_redo_its_analysis(tmp_path, monkeypatch) -> None:
    """A piece that simulates the system needs its rows, not the analysis at (x*, u*) (code
    review, 2026-11-11)."""
    from harness.system import system_of
    path = SYSTEM.run(SYSTEM.Config()).save(tmp_path / "s.json")

    def no_analysis(*_a, **_k):
        raise AssertionError("the analysis ran")

    monkeypatch.setattr("control.frequency_response", no_analysis)
    assert system_of(str(path)).n == 4


@pytest.mark.parametrize("settings,words", [
    ({"x_star": [0.0, 0.0]}, "x_star holds 2 values; the system has 4 states"),
    ({"u_star": [0.0]}, "u_star holds 1 value; the system has 2 inputs"),
    ({"input": 3}, "input = 3: the system has u1…u2"),
    ({"output": 5}, "output = 5: the system has y1…y4"),
    ({"x_star": [float("inf"), 0.0, 0.0, 0.0]}, "is not finite"),
])
def test_an_analysis_request_that_does_not_fit_the_system_is_refused(settings, words) -> None:
    with pytest.raises(ValueError, match=re.escape(words)):
        SYSTEM.run(SYSTEM.Config(**settings))


def test_the_default_run_draws_bode_and_three_responses_with_no_unit() -> None:
    rec = SYSTEM.run(SYSTEM.Config())
    names = [s.name for s in rec.series()]
    assert names == ["bode-magnitude", "bode-phase", "step", "impulse", "ramp"]
    by = {s.name: s for s in rec.series()}
    assert by["bode-magnitude"].x_log and by["bode-magnitude"].y_log and by["bode-phase"].x_log
    assert not by["step"].x_log
    assert all("[" not in s.x_label + s.y_label for s in rec.series())
