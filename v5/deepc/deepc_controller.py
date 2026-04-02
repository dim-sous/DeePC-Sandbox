"""Data-Enabled Predictive Controller (DeePC) — v5.

Robust DeePC with noise-adaptive regularization per
Coulson/Lygeros/Dörfler (arXiv:1903.06804).

Builds on v4's sparse QP form. Key v5 addition: lambda_g and lambda_y
are CVXPY Parameters updated online based on estimated noise level.
Output constraints are tightened proportionally to the noise estimate.
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
    """Receding-horizon DeePC controller (v5) — robust sparse QP form."""

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
        """Construct the sparse QP (called once).

        Decision variables:
            g         — Hankel combination weights (n_cols)
            u         — predicted future inputs (N*m)
            y         — predicted future outputs (N*p)
            sigma_y   — past output slack (Tini*p)
            sigma_out — output constraint slack (N*p, nonneg)

        Linking constraints:
            Uf g = u
            Yf g = y

        Hard constraints:
            Up g = u_ini
            u_lb <= u <= u_ub
            du_lb <= D u <= du_ub
            du_lb <= u[0:m] - u_prev <= du_ub

        Soft constraints (with noise-adaptive tightening):
            Yp g - y_ini = sigma_y
            y_lb + tightening <= y + sigma_out
            y - sigma_out <= y_ub - tightening

        Cost (noise-adaptive regularization):
            ||y - y_ref||²_Q + ||u||²_R
            + lambda_g(t) * ||g||  [L1 or L2]
            + lambda_y(t) * ||σ_y|| [L1 or L2]
            + lambda_out * ||σ_out||²
        """
        cfg = self.config
        n_cols = self.Up.shape[1]

        # Decision variables
        self.g = cp.Variable(n_cols)
        self.u_var = cp.Variable(cfg.N * cfg.m)
        self.y_var = cp.Variable(cfg.N * cfg.p)
        self.sigma_y = cp.Variable(cfg.Tini * cfg.p)
        self.sigma_out = cp.Variable(cfg.N * cfg.p, nonneg=True)

        # Parameters (updated each solve step)
        self.u_ini_param = cp.Parameter(cfg.Tini * cfg.m)
        self.y_ini_param = cp.Parameter(cfg.Tini * cfg.p)
        self.y_ref_param = cp.Parameter(cfg.N * cfg.p)
        self.u_prev_param = cp.Parameter(cfg.m)

        # Noise-adaptive parameters (Wasserstein ball radius scaling)
        self.lambda_g_param = cp.Parameter(nonneg=True, value=cfg.lambda_g)
        self.lambda_y_param = cp.Parameter(nonneg=True, value=cfg.lambda_y)
        self.y_tightening_param = cp.Parameter(
            cfg.N * cfg.p, nonneg=True, value=np.zeros(cfg.N * cfg.p)
        )

        # ── Objective ──────────────────────────────────────────────
        q_sqrt = np.sqrt(np.array(cfg.Q_diag * cfg.N))
        r_sqrt = np.sqrt(np.array(cfg.R_diag * cfg.N))

        tracking_cost = cp.sum_squares(cp.multiply(q_sqrt, self.y_var - self.y_ref_param))
        input_cost = cp.sum_squares(cp.multiply(r_sqrt, self.u_var))

        if cfg.reg_norm_g == "L1":
            reg_g = self.lambda_g_param * cp.norm1(self.g)
        else:
            reg_g = self.lambda_g_param * cp.sum_squares(self.g)

        if cfg.reg_norm_sigma_y == "L1":
            reg_sigma = self.lambda_y_param * cp.norm1(self.sigma_y)
        else:
            reg_sigma = self.lambda_y_param * cp.sum_squares(self.sigma_y)

        reg_out = cfg.lambda_out * cp.sum_squares(self.sigma_out)

        objective = cp.Minimize(
            tracking_cost + input_cost + reg_g + reg_sigma + reg_out
        )

        # ── Constraints ────────────────────────────────────────────
        constraints: list[cp.Constraint] = [
            # Linking: g determines u and y
            self.Uf @ self.g == self.u_var,
            self.Yf @ self.g == self.y_var,
            # Past consistency
            self.Up @ self.g == self.u_ini_param,
            self.Yp @ self.g - self.y_ini_param == self.sigma_y,
        ]

        # Input box constraints (hard)
        u_lb = np.tile([-cfg.delta_max, cfg.a_min], cfg.N)
        u_ub = np.tile([cfg.delta_max, cfg.a_max], cfg.N)
        constraints += [self.u_var >= u_lb, self.u_var <= u_ub]

        # Input rate constraints (hard)
        D = _build_diff_matrix(cfg.N, cfg.m)
        du = D @ self.u_var
        du_lb = np.tile([-cfg.d_delta_max, -cfg.da_max], cfg.N - 1)
        du_ub = np.tile([cfg.d_delta_max, cfg.da_max], cfg.N - 1)
        constraints += [du >= du_lb, du <= du_ub]

        # Cross-solve rate constraint
        du_first = self.u_var[:cfg.m] - self.u_prev_param
        du_first_lb = np.array([-cfg.d_delta_max, -cfg.da_max])
        du_first_ub = np.array([cfg.d_delta_max, cfg.da_max])
        constraints += [du_first >= du_first_lb, du_first <= du_first_ub]

        # Output constraints (soft via sigma_out, with noise-adaptive tightening)
        y_lb_vec = np.tile(cfg.y_lb, cfg.N)
        y_ub_vec = np.tile(cfg.y_ub, cfg.N)

        finite_lb = np.isfinite(y_lb_vec)
        finite_ub = np.isfinite(y_ub_vec)
        if np.any(finite_lb):
            constraints.append(
                self.y_var[finite_lb] >= y_lb_vec[finite_lb]
                + self.y_tightening_param[finite_lb]
                - self.sigma_out[finite_lb]
            )
        if np.any(finite_ub):
            constraints.append(
                self.y_var[finite_ub] <= y_ub_vec[finite_ub]
                - self.y_tightening_param[finite_ub]
                + self.sigma_out[finite_ub]
            )

        self.problem = cp.Problem(objective, constraints)

    def update_robustness(
        self,
        lambda_g: float,
        lambda_y: float,
        tightening: np.ndarray,
    ) -> None:
        """Update noise-adaptive regularization and constraint tightening."""
        self.lambda_g_param.value = lambda_g
        self.lambda_y_param.value = lambda_y
        self.y_tightening_param.value = np.asarray(tightening, dtype=float).ravel()

    def solve(
        self,
        u_ini: np.ndarray,
        y_ini: np.ndarray,
        y_ref: np.ndarray,
        u_prev: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Solve the DeePC optimisation for one control step."""
        cfg = self.config

        u_ini_arr = np.asarray(u_ini, dtype=float)
        self.u_ini_param.value = u_ini_arr.ravel()
        self.y_ini_param.value = np.asarray(y_ini, dtype=float).ravel()
        self.y_ref_param.value = np.asarray(y_ref, dtype=float).ravel()

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
            "lambda_g": float(self.lambda_g_param.value),
            "lambda_y": float(self.lambda_y_param.value),
        }

        if self.g.value is None:
            return np.zeros(cfg.m), info

        info["g_norm"] = float(np.linalg.norm(self.g.value))
        info["sigma_y_norm"] = float(np.linalg.norm(self.sigma_y.value))
        info["sigma_out_norm"] = float(np.linalg.norm(self.sigma_out.value))

        u_predicted = self.u_var.value.reshape(cfg.N, cfg.m)
        y_predicted = self.y_var.value.reshape(cfg.N, cfg.p)
        info["u_predicted"] = u_predicted
        info["y_predicted"] = y_predicted

        u_optimal = u_predicted[0]
        return u_optimal, info
