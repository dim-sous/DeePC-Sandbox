"""Data-Enabled Predictive Controller (DeePC).

Implements the DeePC algorithm from behavioural systems theory.  The
controller uses pre-collected input-output data (via Hankel matrices)
to predict and optimise future system behaviour *without* any explicit
model knowledge.

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


class DeePCController:
    """Receding-horizon DeePC controller.

    The CVXPY optimisation problem is constructed *once* using
    ``cp.Parameter`` objects so that only parameter values are updated
    at each control step (enabling warm-starting and avoiding repeated
    compilation overhead).
    """

    def __init__(
        self,
        config: DeePCConfig,
        u_data: np.ndarray,
        y_data: np.ndarray,
    ) -> None:
        """Initialise the controller from offline data.

        Args:
            config: Experiment configuration.
            u_data: Historical input data, shape (T_data, m).
            y_data: Historical output data, shape (T_data, p).
        """
        self.config = config
        self.u_data = np.atleast_2d(u_data)
        self.y_data = np.atleast_2d(y_data)

        self._build_hankel_matrices()
        self._build_optimization_problem()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_hankel_matrices(self) -> None:
        """Construct and store the four Hankel partitions."""
        cfg = self.config

        # Verify persistent excitation
        is_pe = check_persistent_excitation(
            self.u_data, cfg.L, cfg.m
        )
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
            g       — Hankel combination weights, length n_cols.
            sigma_y — output slack for past-data constraint, length Tini*p.

        Parameters (updated each solve):
            u_ini_param — recent past inputs,  length Tini*m.
            y_ini_param — recent past outputs, length Tini*p.
            y_ref_param — reference trajectory, length N*p.

        Cost:
            ||Y_f g - y_ref||²_Q  +  ||U_f g||²_R
            + λ_g ||g||²  +  λ_y ||σ_y||²

        Constraints:
            U_p g  = u_ini                (hard)
            Y_p g  - y_ini = σ_y          (softened via slack)
            input bounds on U_f g         (box constraints per step)
        """
        cfg = self.config
        n_cols = self.Up.shape[1]

        # Decision variables
        self.g = cp.Variable(n_cols)
        self.sigma_y = cp.Variable(cfg.Tini * cfg.p)

        # Parameters
        self.u_ini_param = cp.Parameter(cfg.Tini * cfg.m)
        self.y_ini_param = cp.Parameter(cfg.Tini * cfg.p)
        self.y_ref_param = cp.Parameter(cfg.N * cfg.p)

        # Predicted future trajectories
        y_future = self.Yf @ self.g   # (N*p,)
        u_future = self.Uf @ self.g   # (N*m,)

        # Weight vectors (sqrt of diagonal entries for weighted sum_squares)
        q_sqrt = np.sqrt(np.array(cfg.Q_diag * cfg.N))  # (N*p,)
        r_sqrt = np.sqrt(np.array(cfg.R_diag * cfg.N))  # (N*m,)

        # Objective — use weighted sum_squares (much faster than quad_form
        # for diagonal weight matrices)
        tracking_cost = cp.sum_squares(
            cp.multiply(q_sqrt, y_future - self.y_ref_param)
        )
        input_cost = cp.sum_squares(cp.multiply(r_sqrt, u_future))
        reg_g = cfg.lambda_g * cp.sum_squares(self.g)
        reg_sigma = cfg.lambda_y * cp.sum_squares(self.sigma_y)

        objective = cp.Minimize(
            tracking_cost + input_cost + reg_g + reg_sigma
        )

        # Constraints
        constraints: list[cp.Constraint] = [
            self.Up @ self.g == self.u_ini_param,
            self.Yp @ self.g - self.y_ini_param == self.sigma_y,
        ]

        # Vectorised input box constraints over the full horizon
        u_lb = np.tile([-cfg.delta_max, cfg.a_min], cfg.N)
        u_ub = np.tile([cfg.delta_max, cfg.a_max], cfg.N)
        constraints += [
            u_future >= u_lb,
            u_future <= u_ub,
        ]

        self.problem = cp.Problem(objective, constraints)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        u_ini: np.ndarray,
        y_ini: np.ndarray,
        y_ref: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Solve the DeePC optimisation for one control step.

        Args:
            u_ini: Recent past inputs,  shape (Tini, m) or flattened.
            y_ini: Recent past outputs, shape (Tini, p) or flattened.
            y_ref: Reference output over the prediction horizon,
                   shape (N, p) or flattened.

        Returns:
            u_optimal: First optimal control input, shape (m,).
            info: Dictionary with solver diagnostics:
                - ``status``: solver status string
                - ``cost``: optimal objective value
                - ``g_norm``: L2 norm of g
                - ``sigma_y_norm``: L2 norm of slack
                - ``u_predicted``: full predicted input trajectory (N, m)
                - ``y_predicted``: full predicted output trajectory (N, p)
        """
        cfg = self.config

        # Set parameter values
        self.u_ini_param.value = np.asarray(u_ini, dtype=float).ravel()
        self.y_ini_param.value = np.asarray(y_ini, dtype=float).ravel()
        self.y_ref_param.value = np.asarray(y_ref, dtype=float).ravel()

        # Solve (with fallback)
        try:
            self.problem.solve(
                solver=cfg.solver,
                verbose=cfg.solver_verbose,
                warm_start=True,
            )
        except (cp.SolverError, Exception):
            self.problem.solve(
                solver="SCS",
                verbose=cfg.solver_verbose,
                warm_start=True,
            )

        # Collect diagnostics
        info: dict = {
            "status": self.problem.status,
            "cost": self.problem.value,
            "g_norm": None,
            "sigma_y_norm": None,
            "u_predicted": None,
            "y_predicted": None,
        }

        if self.g.value is None:
            # Solver failed completely — return zero input as safe fallback
            return np.zeros(cfg.m), info

        g_val = self.g.value
        info["g_norm"] = float(np.linalg.norm(g_val))
        info["sigma_y_norm"] = float(
            np.linalg.norm(self.sigma_y.value)
        )

        u_predicted = (self.Uf @ g_val).reshape(cfg.N, cfg.m)
        y_predicted = (self.Yf @ g_val).reshape(cfg.N, cfg.p)
        info["u_predicted"] = u_predicted
        info["y_predicted"] = y_predicted

        # Receding horizon: apply only the first input
        u_optimal = u_predicted[0]
        return u_optimal, info
