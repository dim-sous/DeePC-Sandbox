"""Every solve is counted by its outcome's own name, every fallback is counted where it happens,
and nothing that failed is scored as if it had an answer (CLAUDE.md, "fail LOUD")."""

from __future__ import annotations

import math

import cvxpy as cp
import numpy as np
import optuna
import pytest

import tune
from deepc.controller import DeePCController, constraints_to_arrays
from plants import PLANT_REGISTRY
from sim.config import build_deepc_config
from sim.simulation import solve_counts
from tests.reference import cases


def _controller():
    """A controller on the frozen coupled-masses training data at the plant's defaults, and one
    solve of it from rest."""
    plant = PLANT_REGISTRY["coupled_masses"]()
    cfg = build_deepc_config(plant)
    data = np.load(cases.REFERENCE_DIR / "coupled_masses-data.npz")
    bounds = constraints_to_arrays(plant.get_constraints(), plant.input_names, plant.output_names)
    ctrl = DeePCController(cfg, data["u_data"], data["y_data"], *bounds)
    return ctrl, lambda: ctrl.solve(np.zeros((cfg.Tini, cfg.m)), np.zeros((cfg.Tini, cfg.p)),
                                    np.zeros((cfg.N, cfg.p)))


def test_an_inaccurate_solve_is_counted_by_its_own_name() -> None:
    """[D2] ``run.py`` and the loop counted a status as optimal when it contained "optimal", so
    OSQP's ``optimal_inaccurate`` counted as optimal. One function counts every status exactly,
    for the loop, the closed-loop piece and the metrics ``tune.py`` scores by."""
    counts = solve_counts({"statuses": ["optimal", "optimal_inaccurate", "optimal"],
                           "solvers": ["OSQP", "OSQP", "SCS"], "zero_input": [False, False, True]})
    assert counts == {"solves_optimal": 2, "solves_optimal_inaccurate": 1, "solves_by_OSQP": 2,
                      "solves_by_SCS": 1, "zero_inputs_applied": 1}


def test_a_solver_error_is_answered_by_scs_and_the_solve_says_so(monkeypatch) -> None:
    """[D3] Any exception from the OSQP solve was retried with SCS, and nothing recorded it. A
    ``SolverError`` still is, and the solve names the solver that answered."""
    ctrl, solve = _controller()
    real = cp.Problem.solve

    def osqp_raises(self, *args, **kwargs):
        if kwargs.get("solver") == "OSQP":
            raise cp.SolverError("probe")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", osqp_raises)
    _u, info = solve()
    assert info["solver"] == "SCS" and not info["no_solution"]


def test_an_exception_other_than_a_solver_error_is_not_retried(monkeypatch) -> None:
    """[D3] ``except (cp.SolverError, Exception)`` retried a programming error with SCS too; any
    exception but a ``SolverError`` now reaches the caller."""
    ctrl, solve = _controller()

    def boom(self, *args, **kwargs):
        raise RuntimeError("probe")

    monkeypatch.setattr(cp.Problem, "solve", boom)
    with pytest.raises(RuntimeError, match="probe"):
        solve()


def test_a_solve_that_leaves_no_solution_is_flagged_where_zeros_are_applied(monkeypatch) -> None:
    """[D3] With no solution the controller returned zeros and the loop applied them, unrecorded.
    The zeros stay, flagged on the solve, and the loop's counts hold them."""
    ctrl, solve = _controller()
    monkeypatch.setattr(cp.Problem, "solve", lambda self, *a, **k: None)
    u, info = solve()
    assert info["no_solution"] and not u.any()
    assert solve_counts({"statuses": [str(info["status"])], "solvers": [info["solver"]],
                         "zero_input": [info["no_solution"]]})["zero_inputs_applied"] == 1


def test_the_excitation_rank_is_kept_not_printed(capsys) -> None:
    """[D3] The excitation check printed a warning and the controller was built anyway; inside
    the pieces the print was swallowed with the loop's output. The rank and the rank it needs
    are kept on the controller for the record (48 of 48 on the frozen coupled-masses data at the
    plant's defaults, seed 42, D13)."""
    ctrl, _solve = _controller()
    assert (ctrl.excitation_rank, ctrl.excitation_rank_required) == (48, 48)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("outcome", ["raises", "nan"])
def test_a_failed_tuning_trial_stops_the_study_rather_than_score_1e6(monkeypatch, outcome) -> None:
    """[D3] ``tune.py`` turned any exception in a trial, and a NaN score, into the score 1e6."""
    def trial_run(*_a, **_k):
        if outcome == "raises":
            raise RuntimeError("probe")
        return {"rmse_e_x1": math.nan}

    monkeypatch.setattr(tune, "run_trial", trial_run)
    trial = optuna.trial.FixedTrial({"Ts": 0.05, "Tini": 4, "N": 20, "T_data": 600,
                                     "lambda_g": 10.0, "lambda_y": 1e4, "reg_norm_g": "L2",
                                     "reg_norm_sigma_y": "L1"})
    expected = (RuntimeError, "probe") if outcome == "raises" else (ValueError, "the study stops")
    with pytest.raises(expected[0], match=expected[1]):
        tune.objective(trial, PLANT_REGISTRY["coupled_masses"](), 1.0, "sinusoidal")
