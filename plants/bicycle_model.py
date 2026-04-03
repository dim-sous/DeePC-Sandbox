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

        if initial_state is not None:
            self.state = np.array(initial_state, dtype=float)
        else:
            self.state = np.array([0.0, 0.0, 0.0, v_default])

        self._initial_state = self.state.copy()

    @property
    def position(self) -> np.ndarray:
        """Absolute position [x, y]."""
        return self.state[:2].copy()

    @property
    def heading(self) -> float:
        return float(self.state[2])

    @property
    def speed(self) -> float:
        return float(self.state[3])

    def step(self, u: np.ndarray) -> np.ndarray:
        """Advance one time step.  Returns the new state [x, y, theta, v]."""
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
        return self.state.copy()

    def reset(self, state: np.ndarray | None = None) -> None:
        """Reset to initial or given state."""
        if state is not None:
            self.state = np.array(state, dtype=float)
        else:
            self.state = self._initial_state.copy()


def compute_path_errors(
    vehicle_state: np.ndarray,
    ref_x: float, ref_y: float, ref_heading: float, ref_v: float,
) -> np.ndarray:
    """Compute error-based outputs [e_lateral, e_heading, e_velocity].

    Parameters
    ----------
    vehicle_state : [x, y, theta, v]
    ref_x, ref_y  : reference path point (nearest waypoint)
    ref_heading   : reference path tangent angle [rad]
    ref_v         : reference speed [m/s]

    Returns
    -------
    [e_lat, e_heading, e_v] where:
        e_lat     = signed lateral (cross-track) error, positive = left of path
        e_heading = heading error (vehicle heading - path heading), wrapped to [-pi, pi]
        e_v       = velocity error (vehicle speed - reference speed)
    """
    x, y, theta, v = vehicle_state

    # Vector from reference point to vehicle
    dx = x - ref_x
    dy = y - ref_y

    # Lateral error: project onto path-normal direction
    e_lat = -dx * np.sin(ref_heading) + dy * np.cos(ref_heading)

    # Heading error, wrapped to [-pi, pi]
    e_heading = theta - ref_heading
    e_heading = (e_heading + np.pi) % (2 * np.pi) - np.pi

    # Velocity error
    e_v = v - ref_v

    return np.array([e_lat, e_heading, e_v])
