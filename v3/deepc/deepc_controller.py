"""Data-Enabled Predictive Controller (DeePC) — v3.

Extends v2 with output scaling for QP conditioning.
Output channels are normalized by their standard deviation before
Hankel construction. The QP operates in scaled space; predictions
are unscaled before returning.

Reference:
    J. Coulson, J. Lygeros, F. Dörfler, "Data-Enabled Predictive
    Control: In the Shallows of the DeePC", 2019.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from config.parameters import DeePCConfig
from deepc.hankel import build_data_matrices
from deepc.regularization import check_persistent_excitation


def _build_diff_matrix(N: int, m: int) -> np.ndarray:
    """Build a first-difference matrix for N steps of m channels."""
    rows = (N - 1) * m
    cols = N * m
    D = np.zeros((rows, cols))
    for k in range(N - 1):
        for j in range(m):
            r = k * m + j
            D[r, k * m + j] = -1.0
            D[r, (k + 1) * m + j] = 1.0
    return D


class DeePCController:
    """Receding-horizon DeePC controller (v3) with output scaling."""

    def __init__(
        self,
        config: DeePCConfig,
        u_data: np.ndarray,
        y_data: np.ndarray,
    ) -> None:
        self.config = config
        self.u_data = np.atleast_2d(u_data)
        self.y_data = np.atleast_2d(y_data)

        self._build_hankel_matrices()
        self._build_optimization_problem()

    def _build_hankel_matrices(self) -> None:
        cfg = self.config
        is_pe = check_persistent_excitation(self.u_data, cfg.L, cfg.m)
        if not is_pe:
            print(
                "WARNING: data does not satisfy PE condition — "
                "DeePC performance may degrade."
            )

        # Output scaling — normalize each channel by its std
        self.y_std = np.std(self.y_data, axis=0)  # (p,)
        self.y_std[self.y_std < 1e-8] = 1.0
        y_data_scaled = self.y_data / self.y_std

        self.Up, self.Yp, self.Uf, self.Yf = build_data_matrices(
            self.u_data, y_data_scaled, cfg.Tini, cfg.N
        )

    def _build_optimization_problem(self) -> None:
        """Construct the parametric CVXPY problem in scaled output space."""
        cfg = self.config
        n_cols = self.Up.shape[1]

        # Decision variables
        self.g = cp.Variable(n_cols)
        self.sigma_y = cp.Variable(cfg.Tini * cfg.p)
        self.sigma_out = cp.Variable(cfg.N * cfg.p, nonneg=True)

        # Parameters (y params will be set in scaled space by solve())
        self.u_ini_param = cp.Parameter(cfg.Tini * cfg.m)
        self.y_ini_param = cp.Parameter(cfg.Tini * cfg.p)
        self.y_ref_param = cp.Parameter(cfg.N * cfg.p)
        self.u_prev_param = cp.Parameter(cfg.m)

        # Predicted trajectories (in scaled space)
        y_future = self.Yf @ self.g  # (N*p,)
        u_future = self.Uf @ self.g  # (N*m,)

        # ── Objective ──────────────────────────────────────────────
        # Q operates in scaled space — all channels have ~unit variance,
        # so Q_diag directly controls relative channel importance
        q_sqrt = np.sqrt(np.array(cfg.Q_diag * cfg.N))
        r_sqrt = np.sqrt(np.array(cfg.R_diag * cfg.N))

        tracking_cost = cp.sum_squares(cp.multiply(q_sqrt, y_future - self.y_ref_param))
        input_cost = cp.sum_squares(cp.multiply(r_sqrt, u_future))

        if cfg.reg_norm_g == "L1":
            reg_g = cfg.lambda_g * cp.norm1(self.g)
        else:
            reg_g = cfg.lambda_g * cp.sum_squares(self.g)

        if cfg.reg_norm_sigma_y == "L1":
            reg_sigma = cfg.lambda_y * cp.norm1(self.sigma_y)
        else:
            reg_sigma = cfg.lambda_y * cp.sum_squares(self.sigma_y)

        reg_out = cfg.lambda_out * cp.sum_squares(self.sigma_out)

        objective = cp.Minimize(
            tracking_cost + input_cost + reg_g + reg_sigma + reg_out
        )

        # ── Constraints ────────────────────────────────────────────
        constraints: list[cp.Constraint] = [
            self.Up @ self.g == self.u_ini_param,
            self.Yp @ self.g - self.y_ini_param == self.sigma_y,
        ]

        # Input box constraints (hard)
        u_lb = np.tile([-cfg.delta_max, cfg.a_min], cfg.N)
        u_ub = np.tile([cfg.delta_max, cfg.a_max], cfg.N)
        constraints += [u_future >= u_lb, u_future <= u_ub]

        # Input rate constraints (hard)
        D = _build_diff_matrix(cfg.N, cfg.m)
        du = D @ u_future
        du_lb = np.tile([-cfg.d_delta_max, -cfg.da_max], cfg.N - 1)
        du_ub = np.tile([cfg.d_delta_max, cfg.da_max], cfg.N - 1)
        constraints += [du >= du_lb, du <= du_ub]

        # Cross-solve rate constraint
        du_first = u_future[:cfg.m] - self.u_prev_param
        du_first_lb = np.array([-cfg.d_delta_max, -cfg.da_max])
        du_first_ub = np.array([cfg.d_delta_max, cfg.da_max])
        constraints += [du_first >= du_first_lb, du_first <= du_first_ub]

        # Output constraints (soft, in scaled space)
        y_lb_scaled = np.tile(np.array(cfg.y_lb) / self.y_std, cfg.N)
        y_ub_scaled = np.tile(np.array(cfg.y_ub) / self.y_std, cfg.N)

        finite_lb = np.isfinite(y_lb_scaled)
        finite_ub = np.isfinite(y_ub_scaled)
        if np.any(finite_lb):
            constraints.append(
                y_future[finite_lb] >= y_lb_scaled[finite_lb] - self.sigma_out[finite_lb]
            )
        if np.any(finite_ub):
            constraints.append(
                y_future[finite_ub] <= y_ub_scaled[finite_ub] + self.sigma_out[finite_ub]
            )

        self.problem = cp.Problem(objective, constraints)

    def solve(
        self,
        u_ini: np.ndarray,
        y_ini: np.ndarray,
        y_ref: np.ndarray,
        u_prev: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Solve the DeePC optimisation for one control step.

        All y-arguments are in original (unscaled) space. Scaling and
        unscaling is handled internally.
        """
        cfg = self.config

        # Scale y inputs
        y_scale_tini = np.tile(self.y_std, cfg.Tini)  # (Tini*p,)
        y_scale_N = np.tile(self.y_std, cfg.N)  # (N*p,)

        u_ini_arr = np.asarray(u_ini, dtype=float)
        self.u_ini_param.value = u_ini_arr.ravel()
        self.y_ini_param.value = np.asarray(y_ini, dtype=float).ravel() / y_scale_tini
        self.y_ref_param.value = np.asarray(y_ref, dtype=float).ravel() / y_scale_N

        if u_prev is not None:
            self.u_prev_param.value = np.asarray(u_prev, dtype=float).ravel()
        else:
            self.u_prev_param.value = u_ini_arr.reshape(-1, cfg.m)[-1]

        solver_kwargs: dict = {
            "solver": cfg.solver,
            "verbose": cfg.solver_verbose,
            "warm_start": True,
        }
        if cfg.solver == "OSQP":
            solver_kwargs.update({
                "eps_abs": 1e-5,
                "eps_rel": 1e-5,
                "max_iter": 10000,
                "polish": True,
            })

        try:
            self.problem.solve(**solver_kwargs)
        except (cp.SolverError, Exception):
            self.problem.solve(
                solver="SCS",
                verbose=cfg.solver_verbose,
                warm_start=True,
            )

        info: dict = {
            "status": self.problem.status,
            "cost": self.problem.value,
            "g_norm": None,
            "sigma_y_norm": None,
            "sigma_out_norm": None,
            "u_predicted": None,
            "y_predicted": None,
        }

        if self.g.value is None:
            return np.zeros(cfg.m), info

        g_val = self.g.value
        info["g_norm"] = float(np.linalg.norm(g_val))

        # Unscale sigma norms for reporting
        sigma_y_scaled = self.sigma_y.value
        sigma_out_scaled = self.sigma_out.value
        info["sigma_y_norm"] = float(np.linalg.norm(sigma_y_scaled * y_scale_tini))
        info["sigma_out_norm"] = float(np.linalg.norm(sigma_out_scaled * y_scale_N))

        u_predicted = (self.Uf @ g_val).reshape(cfg.N, cfg.m)
        # Unscale y predictions
        y_predicted_scaled = (self.Yf @ g_val).reshape(cfg.N, cfg.p)
        y_predicted = y_predicted_scaled * self.y_std

        info["u_predicted"] = u_predicted
        info["y_predicted"] = y_predicted

        u_optimal = u_predicted[0]
        return u_optimal, info
