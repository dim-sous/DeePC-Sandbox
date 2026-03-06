"""Centralized configuration for the DeePC trajectory tracking experiment."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DeePCConfig:
    """All tunable parameters for the DeePC autonomous vehicle experiment."""

    # --- System dimensions ---
    m: int = 2          # number of inputs: [steering angle, acceleration]
    p: int = 3          # number of outputs: [x, y, velocity]

    # --- Simulation ---
    Ts: float = 0.1             # sampling time [s]
    L_wheelbase: float = 2.5    # vehicle wheelbase [m]
    sim_steps: int = 150        # closed-loop control steps
    v_ref: float = 5.0          # reference forward velocity [m/s]

    # --- Data collection ---
    T_data: int = 200                    # number of data samples
    noise_std_output: float = 0.01       # measurement noise std
    input_amplitude_delta: float = 0.3   # steering excitation amplitude [rad]
    input_amplitude_a: float = 1.0       # acceleration excitation amplitude [m/s^2]
    prbs_min_period: int = 3             # min hold time for PRBS

    # --- DeePC horizons ---
    Tini: int = 3       # past / initialization window length
    N: int = 15         # prediction / future horizon

    # --- Cost weights (diagonal entries) ---
    Q_diag: list[float] = field(default_factory=lambda: [10.0, 10.0, 1.0])
    R_diag: list[float] = field(default_factory=lambda: [0.1, 0.1])

    # --- Regularization ---
    lambda_g: float = 100.0     # L2 penalty on g (Hankel weights)
    lambda_y: float = 1e4       # L2 penalty on output slack sigma_y

    # --- Input constraints ---
    delta_max: float = 0.5      # max steering angle [rad]
    a_max: float = 3.0          # max acceleration [m/s^2]
    a_min: float = -5.0         # max braking deceleration [m/s^2]

    # --- Reference trajectory ---
    ref_amplitude: float = 5.0   # sinusoidal lateral amplitude [m]
    ref_frequency: float = 0.05  # sinusoidal frequency [Hz]

    # --- Solver ---
    solver: str = "CLARABEL"
    solver_verbose: bool = False

    @property
    def L(self) -> int:
        """Total Hankel matrix depth: Tini + N."""
        return self.Tini + self.N

    @property
    def Q(self) -> np.ndarray:
        """Diagonal weight vector for outputs, shape (N*p,)."""
        return np.array(self.Q_diag * self.N)

    @property
    def R(self) -> np.ndarray:
        """Diagonal weight vector for inputs, shape (N*m,)."""
        return np.array(self.R_diag * self.N)
