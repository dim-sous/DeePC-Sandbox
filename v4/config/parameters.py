"""Centralized configuration for the v4 DeePC experiment."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DeePCConfig:
    """All tunable parameters for the v4 DeePC experiment."""

    # --- System dimensions ---
    m: int = 2  # number of inputs: [steering angle, acceleration]
    p: int = 3  # number of outputs: [x, y, velocity]

    # --- Simulation ---
    Ts: float = 0.1  # sampling time [s]
    L_wheelbase: float = 2.5  # vehicle wheelbase [m]
    sim_steps: int = 150  # closed-loop control steps
    v_ref: float = 5.0  # reference forward velocity [m/s]

    # --- Data collection ---
    T_data: int = 200  # number of data samples
    noise_std_output: float = 0.01  # measurement noise std
    input_amplitude_delta: float = 0.3  # steering excitation amplitude [rad]
    input_amplitude_a: float = 1.0  # acceleration excitation amplitude [m/s^2]
    prbs_min_period: int = 3  # min hold time for PRBS

    # --- DeePC horizons ---
    Tini: int = 3  # past / initialization window length
    N: int = 15  # prediction / future horizon

    # --- Cost weights (diagonal entries) ---
    Q_diag: list[float] = field(default_factory=lambda: [10.0, 10.0, 1.0])
    R_diag: list[float] = field(default_factory=lambda: [0.1, 0.1])

    # --- Regularization ---
    lambda_g: float = 5.0  # penalty on g
    lambda_y: float = 1e4  # penalty on output slack sigma_y
    reg_norm_g: str = "L2"  # "L1" or "L2" for g regularization
    reg_norm_sigma_y: str = "L1"  # "L1" or "L2" for sigma_y regularization

    # --- Input constraints (hard) ---
    delta_max: float = 0.5  # max steering angle [rad]
    a_max: float = 3.0  # max acceleration [m/s^2]
    a_min: float = -5.0  # max braking deceleration [m/s^2]

    # --- Input rate constraints (hard) ---
    d_delta_max: float = 0.1  # max steering rate [rad/step]
    da_max: float = 0.5  # max acceleration rate (jerk) [m/s^2/step]

    # --- Output constraints (soft) ---
    y_lb: list[float] = field(default_factory=lambda: [float("-inf"), float("-inf"), 0.0])
    y_ub: list[float] = field(default_factory=lambda: [float("inf"), float("inf"), 15.0])
    lambda_out: float = 1e3  # penalty on output slack sigma_out

    # --- Reference trajectory ---
    ref_amplitude: float = 5.0  # sinusoidal lateral amplitude [m]
    ref_frequency: float = 0.05  # sinusoidal frequency [Hz]

    # --- Solver ---
    solver: str = "OSQP"
    solver_verbose: bool = False

    @property
    def L(self) -> int:
        return self.Tini + self.N

    @property
    def Q(self) -> np.ndarray:
        return np.array(self.Q_diag * self.N)

    @property
    def R(self) -> np.ndarray:
        return np.array(self.R_diag * self.N)
