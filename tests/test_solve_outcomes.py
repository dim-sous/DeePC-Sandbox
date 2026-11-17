"""Every solve is counted by its outcome's own name, every fallback is counted where it happens,
and nothing that failed is applied as if it had an answer (CLAUDE.md, "fail LOUD")."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from deepc.config import DeePCConfig
from deepc.controller import DeePCController
from harness import get_component
from harness.closed_loop.run import solve_counts

SYSTEM, OPEN = get_component("system"), get_component("open_loop")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """The default open-loop run on the default system (the two masses), at T = 300."""
    out = tmp_path_factory.mktemp("data")
    system = SYSTEM.run(SYSTEM.Config()).save(out / "masses.json")
    return OPEN.run(OPEN.Config(system_run=str(system), T=300))


def _built(data, rate: float):
    """A controller on that data at the closed-loop piece's default settings, actuator bounds ±5,
    rate bounds ±``rate``, no output bound; and one solve of it from rest."""
    cfg = DeePCConfig(m=2, p=4, Tini=4, N=20, Q_diag=[20.0, 1.0, 20.0, 1.0], R_diag=[0.1, 0.1],
                      lambda_g=10.0, lambda_y=1e4, reg_norm_g="L2", reg_norm_sigma_y="L1")
    inf4 = np.full(4, np.inf)
    ctrl = DeePCController(cfg, data.u, data.y, -5 * np.ones(2), 5 * np.ones(2),
                           -rate * np.ones(2), rate * np.ones(2), -inf4, inf4)
    return ctrl, lambda: ctrl.solve(np.zeros((4, 2)), np.zeros((4, 4)), np.zeros((20, 4)))


@pytest.fixture
def controller(data):
    """Built as the closed-loop piece builds it: no rate bound (±inf)."""
    return _built(data, np.inf)


def test_an_inaccurate_solve_is_counted_by_its_own_name() -> None:
    """[D2] ``run.py`` and the loop counted a status as optimal when it contained "optimal", so
    OSQP's ``optimal_inaccurate`` counted as optimal. One function counts every status exactly,
    and every solver that answered (``harness.closed_loop.run.solve_counts``)."""
    counts = solve_counts(["optimal", "optimal_inaccurate", "optimal"], ["OSQP", "OSQP", "SCS"])
    assert counts == {"solves_optimal": 2, "solves_optimal_inaccurate": 1, "solves_by_OSQP": 2,
                      "solves_by_SCS": 1}


def test_a_solver_error_is_answered_by_scs_and_the_solve_says_so(monkeypatch, data) -> None:
    """[D3] Any exception from the OSQP solve was retried with SCS, and nothing recorded it. A
    ``SolverError`` still is, and the solve names the solver that answered. Built with the old
    coupled masses' rate bound ±2 (``c7f4569:tools/plants/coupled_masses.py``): with the closed
    loop's ±inf, SCS answered ``unbounded_inaccurate`` with no solution after 41 s (D18)."""
    ctrl, solve = _built(data, 2.0)
    real = cp.Problem.solve

    def osqp_raises(self, *args, **kwargs):
        if kwargs.get("solver") == "OSQP":
            raise cp.SolverError("probe")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", osqp_raises)
    _u, info = solve()
    assert info["solver"] == "SCS" and not info["no_solution"]


def test_an_exception_other_than_a_solver_error_is_not_retried(monkeypatch, controller) -> None:
    """[D3] ``except (cp.SolverError, Exception)`` retried a programming error with SCS too; any
    exception but a ``SolverError`` now reaches the caller."""
    ctrl, solve = controller

    def boom(self, *args, **kwargs):
        raise RuntimeError("probe")

    monkeypatch.setattr(cp.Problem, "solve", boom)
    with pytest.raises(RuntimeError, match="probe"):
        solve()


def test_a_solve_that_leaves_no_solution_is_flagged_where_zeros_are_applied(monkeypatch, controller) -> None:
    """[D3] With no solution the controller returned zeros and the loop applied them, unrecorded.
    The zeros stay in the controller, flagged on the solve; the closed-loop piece stops on the
    flag (``tests/test_closed_loop.py``)."""
    ctrl, solve = controller
    monkeypatch.setattr(cp.Problem, "solve", lambda self, *a, **k: None)
    u, info = solve()
    assert info["no_solution"] and not u.any()


def test_the_excitation_rank_is_kept_not_printed(capsys, controller) -> None:
    """[D3] The excitation check printed a warning and the controller was built anyway; inside
    the pieces the print was swallowed with the loop's output. The rank and the rank it needs
    are kept on the controller for the record (48 of 48, L = 24 and m = 2, on the default
    open-loop run at T = 300, seed 42; D13)."""
    ctrl, _solve = controller
    assert (ctrl.excitation_rank, ctrl.excitation_rank_required) == (48, 48)
    assert capsys.readouterr().out == ""
