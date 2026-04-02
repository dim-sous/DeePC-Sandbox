"""Data-Enabled Predictive Controller (DeePC).

Sparse QP form with optional online sliding Hankel window and
noise-adaptive regularization. The Hankel matrices (Up, Yp, Uf, Yf)
are CVXPY Parameters updated each control step with fresh closed-loop
data. The QP structure never changes — warm-start preserved.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from control.config import DeePCConfig
from control.online_hankel import SlidingHankelWindow
from control.regularization import check_persistent_excitation


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
    """Receding-horizon DeePC controller — sparse QP with online Hankel."""

    def __init__(
        self,
        config: DeePCConfig,
        u_data: np.ndarray,
        y_data: np.ndarray,
    ) -> None:
        self.config = config

        u_data = np.atleast_2d(u_data)
        is_pe = check_persistent_excitation(u_data, config.L, config.m)
        if not is_pe:
            print(
                "WARNING: data does not satisfy PE condition — "
                "DeePC performance may degrade."
            )

        self._hankel = SlidingHankelWindow(
            u_data, np.atleast_2d(y_data),
            config.Tini, config.N, config.m, config.p,
            config.hankel_window_size,
        )

        self._build_optimization_problem()

    def _build_optimization_problem(self) -> None:
        """Construct the sparse QP with Parameter Hankel matrices."""
        cfg = self.config
        n_cols = self._hankel.n_cols

        # Decision variables
        self.g = cp.Variable(n_cols)
        self.u_var = cp.Variable(cfg.N * cfg.m)
        self.y_var = cp.Variable(cfg.N * cfg.p)
        self.sigma_y = cp.Variable(cfg.Tini * cfg.p)
        self.sigma_out = cp.Variable(cfg.N * cfg.p, nonneg=True)

        # Scalar / vector parameters
        self.u_ini_param = cp.Parameter(cfg.Tini * cfg.m)
        self.y_ini_param = cp.Parameter(cfg.Tini * cfg.p)
        self.y_ref_param = cp.Parameter(cfg.N * cfg.p)
        self.u_prev_param = cp.Parameter(cfg.m)

        # Noise-adaptive parameters
        self.lambda_g_param = cp.Parameter(nonneg=True, value=cfg.lambda_g)
        self.lambda_y_param = cp.Parameter(nonneg=True, value=cfg.lambda_y)
        self.y_tightening_param = cp.Parameter(
            cfg.N * cfg.p, nonneg=True, value=np.zeros(cfg.N * cfg.p)
        )

        # Hankel matrices as CVXPY Parameters
        self.Up_param = cp.Parameter(
            (cfg.Tini * cfg.m, n_cols), value=self._hankel.Up
        )
        self.Yp_param = cp.Parameter(
            (cfg.Tini * cfg.p, n_cols), value=self._hankel.Yp
        )
        self.Uf_param = cp.Parameter(
            (cfg.N * cfg.m, n_cols), value=self._hankel.Uf
        )
        self.Yf_param = cp.Parameter(
            (cfg.N * cfg.p, n_cols), value=self._hankel.Yf
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
            self.Uf_param @ self.g == self.u_var,
            self.Yf_param @ self.g == self.y_var,
            self.Up_param @ self.g == self.u_ini_param,
            self.Yp_param @ self.g - self.y_ini_param == self.sigma_y,
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

    def update_hankel(self, u_new: np.ndarray, y_new: np.ndarray) -> dict:
        """Slide the Hankel window and push new values to CVXPY."""
        info = self._hankel.update(u_new, y_new)
        if info.get("updated"):
            self.Up_param.value = self._hankel.Up
            self.Yp_param.value = self._hankel.Yp
            self.Uf_param.value = self._hankel.Uf
            self.Yf_param.value = self._hankel.Yf
        return info

    def update_robustness(
        self,
        lambda_g: float,
        lambda_y: float,
        tightening: np.ndarray,
    ) -> None:
        """Update noise-adaptive regularization."""
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
            "hankel_updates": self._hankel.n_online_updates,
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
