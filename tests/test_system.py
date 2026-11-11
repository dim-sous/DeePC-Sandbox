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
    assert (plant.n, plant.m, plant.p, plant.q) == (4, 2, 4, 4)
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
    for u in U:
        x, exact = hold(plant.f, x, u, (0.0, Ts))[1][-1], Ad @ exact + Bd @ u
        worst = max(worst, float(np.abs(x - exact).max()))
    assert (RTOL, ATOL) == (1e-9, 1e-12) and worst < 1e-8, worst


def test_the_integrator_matches_closed_forms() -> None:
    decay = parse(["-theta1*x1 + u1"], ["x1"], [1.0])
    t, x = hold(decay.f, np.array([2.0]), np.zeros(1), (0.0, 3.0))
    assert np.allclose(x[:, 0], 2.0 * np.exp(-t), rtol=1e-8, atol=1e-11)
    ramp = parse(["u1"], ["x1"], [])
    assert hold(ramp.f, np.array([1.0]), np.array([0.5]), (0.0, 2.0))[1][-1] == \
        pytest.approx([2.0], abs=1e-11)


def test_the_system_run_records_its_rows_as_read_and_its_sizes(tmp_path) -> None:
    rec = SYSTEM.run(SYSTEM.Config())
    assert rec.numbers == {"n": 4, "m": 2, "p": 4, "q": 4}
    assert rec.constants["h_read"] == ["x1", "x2", "x3", "x4"] and rec.series() == ()
    again = SYSTEM.Record.load(rec.save(tmp_path / "s.json"))
    assert again.config.f == rec.config.f and again.numbers == rec.numbers


def test_a_list_of_rows_round_trips_through_the_command_line_words() -> None:
    cfg = with_overrides(SYSTEM.Config(), [("f", '["x2", "-sin(x1) - theta1*x2 + u1"]'),
                                           ("h", '["x1"]'), ("theta", "[0.2]")])
    assert cfg.f == ["x2", "-sin(x1) - theta1*x2 + u1"] and cfg.theta == [0.2]
    assert SYSTEM.run(cfg).numbers == {"n": 2, "m": 1, "p": 1, "q": 1}
    with pytest.raises(ValueError, match="is not a JSON list of texts"):
        with_overrides(SYSTEM.Config(), [("f", "x2, -x1 + u1")])
