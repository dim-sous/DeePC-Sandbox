"""Nonlinear kinematic bicycle model.

A simple 2-D vehicle plant useful as a first benchmark for DeePC
controllers.  The model is intentionally decoupled from any specific
DeePC configuration so it can be reused across versions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plants.base import Constraints, DataCollectionConfig, PlantBase


# ── Standalone helper (kept for backward compatibility) ───────────────

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

    dx = x - ref_x
    dy = y - ref_y

    e_lat = -dx * np.sin(ref_heading) + dy * np.cos(ref_heading)

    e_heading = theta - ref_heading
    e_heading = (e_heading + np.pi) % (2 * np.pi) - np.pi

    e_v = v - ref_v

    return np.array([e_lat, e_heading, e_v])


# ── Bicycle model ────────────────────────────────────────────────────

class BicycleModel(PlantBase):
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
        Ts: float = 0.1,
        L_wheelbase: float = 2.5,
        delta_max: float = 0.5,
        a_max: float = 1.0,
        a_min: float = -1.0,
        d_delta_max: float = 0.3,
        da_max: float = 0.5,
        v_ref: float = 5.0,
        ref_amplitude: float = 5.0,
        ref_frequency: float = 0.05,
        input_amplitude_delta: float = 0.5,
        input_amplitude_a: float = 2.5,
        prbs_min_period: int = 3,
        initial_state: np.ndarray | None = None,
        v_default: float = 5.0,
    ) -> None:
        self._Ts = Ts
        self.L_wheelbase = L_wheelbase
        self.delta_max = delta_max
        self.a_max = a_max
        self.a_min = a_min
        self.d_delta_max = d_delta_max
        self.da_max = da_max
        self.v_ref = v_ref
        self.ref_amplitude = ref_amplitude
        self.ref_frequency = ref_frequency
        self.input_amplitude_delta = input_amplitude_delta
        self.input_amplitude_a = input_amplitude_a
        self.prbs_min_period = prbs_min_period
        self.v_default = v_default

        if initial_state is not None:
            self._state = np.array(initial_state, dtype=float)
        else:
            self._state = np.array([0.0, 0.0, 0.0, v_default])

        self._initial_state = self._state.copy()

    # ── PlantBase required properties ─────────────────────────────────

    @property
    def m(self) -> int:
        return 2

    @property
    def p(self) -> int:
        return 3

    @property
    def Ts(self) -> float:
        return self._Ts

    @property
    def input_names(self) -> list[str]:
        return ["steering", "acceleration"]

    @property
    def output_names(self) -> list[str]:
        return ["e_lateral", "e_heading", "e_velocity"]

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    # ── Convenience properties ────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        """Absolute position [x, y]."""
        return self._state[:2].copy()

    @property
    def heading(self) -> float:
        return float(self._state[2])

    @property
    def speed(self) -> float:
        return float(self._state[3])

    # ── PlantBase required methods ────────────────────────────────────

    def step(self, u: np.ndarray) -> np.ndarray:
        """Advance one time step.  Returns the new state [x, y, theta, v]."""
        delta = np.clip(u[0], -self.delta_max, self.delta_max)
        a = np.clip(u[1], self.a_min, self.a_max)

        x, y, theta, v = self._state
        Ts = self._Ts
        Lw = self.L_wheelbase

        x_new = x + v * np.cos(theta) * Ts
        y_new = y + v * np.sin(theta) * Ts
        theta_new = theta + (v / Lw) * np.tan(delta) * Ts
        v_new = max(v + a * Ts, 0.0)

        self._state = np.array([x_new, y_new, theta_new, v_new])
        return self._state.copy()

    def reset(self, state: np.ndarray | None = None) -> None:
        """Reset to initial or given state."""
        if state is not None:
            self._state = np.array(state, dtype=float)
            self._initial_state = self._state.copy()
        else:
            self._state = self._initial_state.copy()

    def get_output(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Compute [e_lateral, e_heading, e_velocity] relative to reference.

        *reference* is ``[x, y, heading, v]``.
        """
        return compute_path_errors(state, *reference)

    def get_constraints(self) -> Constraints:
        return Constraints(
            u_lb={"steering": -self.delta_max, "acceleration": self.a_min},
            u_ub={"steering": self.delta_max, "acceleration": self.a_max},
            du_min={"steering": -self.d_delta_max, "acceleration": -self.da_max},
            du_max={"steering": self.d_delta_max, "acceleration": self.da_max},
            y_lb={"e_lateral": float("-inf"), "e_heading": float("-inf"),
                   "e_velocity": 0.0},
            y_ub={"e_lateral": float("inf"), "e_heading": float("inf"),
                   "e_velocity": 10.0},
        )

    def get_default_config_overrides(self) -> dict[str, Any]:
        return {
            "Q_diag": [10.0, 10.0, 0.0],
            "R_diag": [0.1, 0.1],
            "T_data": 400,
            "noise_std_output": 0.01,
            "Tini": 3,
            "N": 15,
            "sim_duration": 60.0,
            "lambda_g": 5.0,
            "lambda_y": 1e4,
            "lambda_out": 0.0,
            "reg_norm_g": "L2",
            "reg_norm_sigma_y": "L1",
        }

    def get_scenarios(self) -> dict[str, Any]:
        return {
            "default": self._generate_sinusoidal_path,
            "lissajous": self._generate_lissajous_path,
        }

    def get_data_collection_config(self) -> DataCollectionConfig:
        speed_offsets = [0.0, -0.4, 0.4, -0.7, 0.7]
        initial_conditions = []
        for dv in speed_offsets:
            initial_conditions.append({"speed_offset": dv})

        return DataCollectionConfig(
            initial_conditions=initial_conditions,
            excitation_fn=self._generate_excitation,
            stabilizing_controller=self._stabilizing_controller,
            nominal_reference=self._nominal_reference,
        )

    # ── Optional hooks ────────────────────────────────────────────────

    def get_initial_state_for_scenario(
        self, first_waypoint: np.ndarray,
    ) -> np.ndarray:
        """Derive initial state [x, y, heading, v] from first reference."""
        return first_waypoint

    def get_position_from_state(self, state: np.ndarray) -> np.ndarray | None:
        return state[:2].copy()

    def get_tuning_objective_key(self) -> str:
        return "rmse_position"

    def compute_custom_metrics(self, results: dict) -> dict[str, float]:
        metrics: dict[str, float] = {}
        pos = np.asarray(results.get("pos_history", []))
        ref_pos = np.asarray(results.get("ref_pos_history", []))
        if len(pos) > 0 and len(ref_pos) > 0:
            n = min(len(pos), len(ref_pos))
            pos_err = pos[:n] - ref_pos[:n]
            metrics["rmse_x"] = float(np.sqrt(np.mean(pos_err[:, 0] ** 2)))
            metrics["rmse_y"] = float(np.sqrt(np.mean(pos_err[:, 1] ** 2)))
            metrics["rmse_position"] = float(
                np.sqrt(np.mean(pos_err[:, 0] ** 2 + pos_err[:, 1] ** 2))
            )
        return metrics

    # ── Scenario generators ───────────────────────────────────────────

    def _generate_sinusoidal_path(
        self, Tini: int, N: int, sim_steps: int, Ts: float,
    ) -> np.ndarray:
        """Sinusoidal reference path (default scenario)."""
        total = Tini + sim_steps + N
        A = self.ref_amplitude
        f = self.ref_frequency

        path = np.zeros((total, 4))
        for k in range(total):
            t = k * Ts
            path[k, 0] = self.v_ref * t + A * np.sin(2 * np.pi * f * t)
            path[k, 1] = A * np.sin(2 * np.pi * 2 * f * t)

        for k in range(total - 1):
            dx = path[k + 1, 0] - path[k, 0]
            dy = path[k + 1, 1] - path[k, 1]
            path[k, 2] = np.arctan2(dy, dx)
            path[k, 3] = np.sqrt(dx ** 2 + dy ** 2) / Ts
        path[-1, 2] = path[-2, 2]
        path[-1, 3] = path[-2, 3]

        return path

    def _generate_lissajous_path(
        self, Tini: int, N: int, sim_steps: int, Ts: float,
    ) -> np.ndarray:
        """Lissajous figure-8 reference path."""
        total = Tini + sim_steps + N
        A = self.ref_amplitude
        f = self.ref_frequency

        path = np.zeros((total, 4))
        for k in range(total):
            t = k * Ts
            path[k, 0] = A * np.sin(2 * np.pi * f * t)
            path[k, 1] = A * np.sin(2 * np.pi * 2 * f * t)

        for k in range(total - 1):
            dx = path[k + 1, 0] - path[k, 0]
            dy = path[k + 1, 1] - path[k, 1]
            path[k, 2] = np.arctan2(dy, dx)
            path[k, 3] = np.sqrt(dx ** 2 + dy ** 2) / Ts
        path[-1, 2] = path[-2, 2]
        path[-1, 3] = path[-2, 3]

        return path

    # ── Data collection helpers ───────────────────────────────────────

    def _generate_excitation(
        self, T_per: int, rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate (T_per, 2) excitation signals for [steering, accel]."""
        from plants.excitation import generate_multisine, generate_prbs

        # Steering: PRBS + uniform random
        delta_prbs = generate_prbs(
            T_per, self.input_amplitude_delta, self.prbs_min_period, rng,
        )
        delta_rand = rng.uniform(
            -self.input_amplitude_delta, self.input_amplitude_delta, size=T_per,
        )
        delta_exc = 0.5 * delta_prbs + 0.5 * delta_rand

        # Acceleration: multisine + uniform random
        a_sine = generate_multisine(
            T_per, self.input_amplitude_a, n_freqs=10, Ts=self._Ts, rng=rng,
        )
        a_rand = rng.uniform(
            -self.input_amplitude_a, self.input_amplitude_a, size=T_per,
        )
        a_exc = 0.5 * a_sine + 0.5 * a_rand

        return np.column_stack([delta_exc, a_exc])

    def _stabilizing_controller(self, errors: np.ndarray) -> np.ndarray:
        """Proportional stabilizing baseline for data collection."""
        K_lat = 0.3
        K_head = 1.0
        K_v = 0.5
        return np.array([
            -K_lat * errors[0] - K_head * errors[1],
            -K_v * errors[2],
        ])

    def _nominal_reference(self, k: int, Ts: float) -> np.ndarray:
        """Straight-line nominal reference at v_ref for data collection."""
        nom_x = (k + 1) * self.v_ref * Ts
        return np.array([nom_x, 0.0, 0.0, self.v_ref])

    def make_episode_initial_state(
        self, condition: dict[str, Any], rng: np.random.Generator,
    ) -> np.ndarray:
        """Create an initial state for one data-collection episode."""
        dv = condition["speed_offset"]
        v_init = max(self.v_ref + dv * self.v_ref, 0.5)
        heading_init = rng.uniform(-0.3, 0.3)
        return np.array([0.0, 0.0, heading_init, v_init])

    # ── Units, from the class docstring ──────────────────────────────

    @property
    def input_units(self) -> list[str]:
        return ["rad", "m/s²"]

    @property
    def output_units(self) -> list[str]:
        return ["m", "rad", "m/s"]

    @property
    def metric_units(self) -> dict[str, str]:
        return {"rmse_x": "m", "rmse_y": "m", "rmse_position": "m"}
