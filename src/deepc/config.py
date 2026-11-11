"""Centralized configuration for the DeePC algorithm.

Contains only algorithm-level parameters.  Plant-specific settings
(dimensions, constraints, reference params) live on the plant object
and flow into the config on the bench, which builds it from the plant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class DeePCConfig:
    """All tunable parameters for the DeePC algorithm."""

    # --- System dimensions (set from plant at construction) ---
    m: int = 2
    p: int = 3
    Ts: float = 0.1

    # --- Simulation ---
    sim_duration: float = 60.0

    # --- Data collection (algorithm-level) ---
    T_data: int = 400
    noise_std_output: float = 0.01

    # --- DeePC horizons ---
    Tini: int = 3
    N: int = 15

    # --- Cost weights (diagonal entries) ---
    Q_diag: list[float] = field(default_factory=lambda: [10.0, 10.0, 0.0])
    R_diag: list[float] = field(default_factory=lambda: [0.1, 0.1])

    # --- Regularization ---
    lambda_g: float = 5.0
    lambda_y: float = 1e4
    reg_norm_g: str = "L2"
    reg_norm_sigma_y: str = "L1"

    # --- Output constraint slack ---
    lambda_out: float = 0.0

    # --- Robust DeePC (noise-adaptive regularization) ---
    noise_estimation_window: int = 10
    baseline_noise_std: float = 0.01
    max_lambda_scaling: float = 10.0
    constraint_tightening_factor: float = 2.0

    # --- Online Hankel window ---
    hankel_window_size: int = 0
    hankel_warmup_steps: int = 100

    # --- Solver ---
    solver: str = "OSQP"
    solver_verbose: bool = False

    @property
    def sim_steps(self) -> int:
        return int(self.sim_duration / self.Ts)

    @property
    def L(self) -> int:
        return self.Tini + self.N

    @property
    def Q(self) -> np.ndarray:
        return np.array(self.Q_diag * self.N)

    @property
    def R(self) -> np.ndarray:
        return np.array(self.R_diag * self.N)

