"""Nonlinear kinematic bicycle model.

A simple 2-D vehicle plant useful as a first benchmark for DeePC
controllers.  The model is intentionally decoupled from any specific
DeePC configuration so it can be reused across versions.
"""

import numpy as np


class BicycleModel:
    """Kinematic bicycle model (nonlinear, discrete-time).

    State vector:  [x, y, theta, v]
        x, y   -- position in the global frame [m]
        theta  -- heading angle [rad]
        v      -- longitudinal velocity [m/s]

    Input vector:  [delta, a]
        delta  -- front steering angle [rad]
        a      -- longitudinal acceleration [m/s^2]

    Output vector: [x, y, v]
        The heading is *not* exposed as an output; the controller
        must work with position and speed only.

    Dynamics (forward Euler):
        x(k+1)     = x(k) + v(k) * cos(theta(k)) * Ts
        y(k+1)     = y(k) + v(k) * sin(theta(k)) * Ts
        theta(k+1) = theta(k) + (v(k) / L_wb) * tan(delta(k)) * Ts
        v(k+1)     = v(k) + a(k) * Ts
    """

    def __init__(
        self,
        Ts: float,
        L_wheelbase: float,
        delta_max: float,
        a_max: float,
        a_min: float,
        initial_state: np.ndarray | None = None,
        v_default: float = 5.0,
    ) -> None:
        self.Ts = Ts
        self.L_wheelbase = L_wheelbase
        self.delta_max = delta_max
        self.a_max = a_max
        self.a_min = a_min
        self._nx = 4  # internal state dimension

        if initial_state is not None:
            self.state = np.array(initial_state, dtype=float)
        else:
            self.state = np.array([0.0, 0.0, 0.0, v_default])

        self._initial_state = self.state.copy()

    @property
    def output(self) -> np.ndarray:
        """Current measurement vector [x, y, v]."""
        return self.state[np.array([0, 1, 3])]

    def step(self, u: np.ndarray) -> np.ndarray:
        """Advance one time step and return the new output.

        Args:
            u: Control input [delta, a], shape (m,).

        Returns:
            Measurement [x, y, v], shape (p,).
        """
        delta = np.clip(u[0], -self.delta_max, self.delta_max)
        a = np.clip(u[1], self.a_min, self.a_max)

        x, y, theta, v = self.state
        Ts = self.Ts
        Lw = self.L_wheelbase

        x_new = x + v * np.cos(theta) * Ts
        y_new = y + v * np.sin(theta) * Ts
        theta_new = theta + (v / Lw) * np.tan(delta) * Ts
        v_new = max(v + a * Ts, 0.0)

        self.state = np.array([x_new, y_new, theta_new, v_new])
        return self.output

    def reset(self, state: np.ndarray | None = None) -> None:
        """Reset to initial or given state."""
        if state is not None:
            self.state = np.array(state, dtype=float)
        else:
            self.state = self._initial_state.copy()
