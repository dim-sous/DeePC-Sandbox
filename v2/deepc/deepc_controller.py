"""Data-Enabled Predictive Controller (DeePC) — v2.

Extends v1 with:
  - Input rate constraints (hard)
  - Output constraints (soft, via slack sigma_out)
  - Configurable L1/L2 regularization on g and sigma_y

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
    """Build a first-difference matrix for N steps of m channels.

    Returns D of shape ((N-1)*m, N*m) such that D @ u_flat gives
    [u(1)-u(0), u(2)-u(1), ...] with all m channels interleaved.
    """
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
    """Receding-horizon DeePC controller (v2)."""

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
        self.Up, self.Yp, self.Uf, self.Yf = build_data_matrices(
            self.u_data, self.y_data, cfg.Tini, cfg.N
        )

    def _build_optimization_problem(self) -> None:
        """Construct the parametric CVXPY problem (called once).

        Decision variables:
            g         — Hankel combination weights, length n_cols
            sigma_y   — output slack for past-data constraint, length Tini*p
            sigma_out — output constraint slack, length N*p (nonneg)

        Hard constraints:
            U_p g = u_ini
            u_lb <= U_f g <= u_ub
            du_lb <= Delta(U_f g) <= du_ub

        Soft constraints:
            Y_p g - y_ini = sigma_y
            y_lb <= Y_f g + sigma_out, Y_f g - sigma_out <= y_ub

        Cost:
            ||Y_f g - y_ref||^2_Q + ||U_f g||^2_R
            + lambda_g * norm(g)         [L1 or L2]
            + lambda_y * norm(sigma_y)   [L1 or L2]
            + lambda_out * ||sigma_out||^2
        """
        cfg = self.config
        n_cols = self.Up.shape[1]

        # Decision variables
        self.g = cp.Variable(n_cols)
        self.sigma_y = cp.Variable(cfg.Tini * cfg.p)
        self.sigma_out = cp.Variable(cfg.N * cfg.p, nonneg=True)

        # Parameters
        self.u_ini_param = cp.Parameter(cfg.Tini * cfg.m)
        self.y_ini_param = cp.Parameter(cfg.Tini * cfg.p)
        self.y_ref_param = cp.Parameter(cfg.N * cfg.p)
        self.u_prev_param = cp.Parameter(cfg.m)  # last applied input (for rate constraint)

        # Predicted trajectories
        y_future = self.Yf @ self.g  # (N*p,)
        u_future = self.Uf @ self.g  # (N*m,)

        # ── Objective ──────────────────────────────────────────────
        q_sqrt = np.sqrt(np.array(cfg.Q_diag * cfg.N))
        r_sqrt = np.sqrt(np.array(cfg.R_diag * cfg.N))

        tracking_cost = cp.sum_squares(cp.multiply(q_sqrt, y_future - self.y_ref_param))
        input_cost = cp.sum_squares(cp.multiply(r_sqrt, u_future))

        # Regularization — L1 or L2
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
            # Past consistency
            self.Up @ self.g == self.u_ini_param,
            self.Yp @ self.g - self.y_ini_param == self.sigma_y,
        ]

        # Input box constraints (hard)
        u_lb = np.tile([-cfg.delta_max, cfg.a_min], cfg.N)
        u_ub = np.tile([cfg.delta_max, cfg.a_max], cfg.N)
        constraints += [u_future >= u_lb, u_future <= u_ub]

        # Input rate constraints (hard)
        # Within-horizon rates: u(k+1) - u(k) for k=0..N-2
        D = _build_diff_matrix(cfg.N, cfg.m)
        du = D @ u_future
        du_lb = np.tile([-cfg.d_delta_max, -cfg.da_max], cfg.N - 1)
        du_ub = np.tile([cfg.d_delta_max, cfg.da_max], cfg.N - 1)
        constraints += [du >= du_lb, du <= du_ub]

        # Cross-solve rate: u_future[0] - u_prev
        du_first = u_future[:cfg.m] - self.u_prev_param
        du_first_lb = np.array([-cfg.d_delta_max, -cfg.da_max])
        du_first_ub = np.array([cfg.d_delta_max, cfg.da_max])
        constraints += [du_first >= du_first_lb, du_first <= du_first_ub]

        # Output constraints (soft via sigma_out)
        y_lb_vec = np.tile(cfg.y_lb, cfg.N)
        y_ub_vec = np.tile(cfg.y_ub, cfg.N)

        # Only add constraints for finite bounds
        finite_lb = np.isfinite(y_lb_vec)
        finite_ub = np.isfinite(y_ub_vec)
        if np.any(finite_lb):
            constraints.append(
                y_future[finite_lb] >= y_lb_vec[finite_lb] - self.sigma_out[finite_lb]
            )
        if np.any(finite_ub):
            constraints.append(
                y_future[finite_ub] <= y_ub_vec[finite_ub] + self.sigma_out[finite_ub]
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

        Args:
            u_ini: Recent past inputs, shape (Tini, m).
            y_ini: Recent past outputs, shape (Tini, p).
            y_ref: Reference over prediction horizon, shape (N, p).
            u_prev: Last applied input, shape (m,). Used for cross-solve
                    rate constraint. Defaults to last row of u_ini.

        Returns:
            u_optimal: First optimal control input, shape (m,).
            info: Dictionary with solver diagnostics.
        """
        cfg = self.config

        u_ini_arr = np.asarray(u_ini, dtype=float)
        self.u_ini_param.value = u_ini_arr.ravel()
        self.y_ini_param.value = np.asarray(y_ini, dtype=float).ravel()
        self.y_ref_param.value = np.asarray(y_ref, dtype=float).ravel()

        if u_prev is not None:
            self.u_prev_param.value = np.asarray(u_prev, dtype=float).ravel()
        else:
            # Default: last row of u_ini
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
        info["sigma_y_norm"] = float(np.linalg.norm(self.sigma_y.value))
        info["sigma_out_norm"] = float(np.linalg.norm(self.sigma_out.value))

        u_predicted = (self.Uf @ g_val).reshape(cfg.N, cfg.m)
        y_predicted = (self.Yf @ g_val).reshape(cfg.N, cfg.p)
        info["u_predicted"] = u_predicted
        info["y_predicted"] = y_predicted

        u_optimal = u_predicted[0]
        return u_optimal, info
