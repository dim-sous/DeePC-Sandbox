r"""Coupled two-mass spring-damper system (LTI MIMO).

A pair of masses connected by springs and dampers, each driven by an
external force.  A standard benchmark for MIMO controller design.

    ┌─────┐   k12, c12    ┌─────┐
    │ m1  ├───/\/\/───────┤ m2  │
    └──┬──┘               └──┬──┘
    k1,c1 (wall)          k2,c2 (wall)
       │                     │
     ─────                 ─────

State vector:   [x1, v1, x2, v2]
Input vector:   [F1, F2]
Output vector:  [e_x1, e_v1, e_x2, e_v2]  (errors vs reference)

Continuous dynamics:
    m1 * x1'' = -k1*x1 - c1*v1 + k12*(x2-x1) + c12*(v2-v1) + F1
    m2 * x2'' = -k2*x2 - c2*v2 + k12*(x1-x2) + c12*(v1-v2) + F2

Discretised with forward Euler at sampling time Ts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plants.base import Constraints, DataCollectionConfig, PlantBase


class CoupledMasses(PlantBase):
    """Discrete-time coupled two-mass spring-damper (LTI MIMO)."""

    def __init__(
        self,
        Ts: float = 0.05,
        m1: float = 1.0,
        m2: float = 1.0,
        k1: float = 2.0,
        k2: float = 2.0,
        k12: float = 1.0,
        c1: float = 0.3,
        c2: float = 0.3,
        c12: float = 0.1,
        F_max: float = 5.0,
        dF_max: float = 2.0,
        initial_state: np.ndarray | None = None,
    ) -> None:
        self._Ts = Ts
        self._m1 = m1
        self._m2 = m2
        self.F_max = F_max
        self.dF_max = dF_max

        # Build continuous A, B
        #   state = [x1, v1, x2, v2]
        Ac = np.array([
            [0,          1,          0,          0         ],
            [-(k1+k12)/m1, -(c1+c12)/m1, k12/m1,    c12/m1   ],
            [0,          0,          0,          1         ],
            [k12/m2,     c12/m2,     -(k2+k12)/m2, -(c2+c12)/m2],
        ])
        Bc = np.array([
            [0,     0    ],
            [1/m1,  0    ],
            [0,     0    ],
            [0,     1/m2 ],
        ])

        # Forward Euler discretisation
        self._A = np.eye(4) + Ts * Ac
        self._B = Ts * Bc

        if initial_state is not None:
            self._state = np.array(initial_state, dtype=float)
        else:
            self._state = np.zeros(4)
        self._initial_state = self._state.copy()

    # ── PlantBase required properties ─────────────────────────────────

    @property
    def m(self) -> int:
        return 2

    @property
    def p(self) -> int:
        return 4

    @property
    def Ts(self) -> float:
        return self._Ts

    @property
    def input_names(self) -> list[str]:
        return ["F1", "F2"]

    @property
    def output_names(self) -> list[str]:
        return ["e_x1", "e_v1", "e_x2", "e_v2"]

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    # ── PlantBase required methods ────────────────────────────────────

    def step(self, u: np.ndarray) -> np.ndarray:
        u_clipped = np.clip(u, -self.F_max, self.F_max)
        self._state = self._A @ self._state + self._B @ u_clipped
        return self._state.copy()

    def reset(self, state: np.ndarray | None = None) -> None:
        if state is not None:
            self._state = np.array(state, dtype=float)
            self._initial_state = self._state.copy()
        else:
            self._state = self._initial_state.copy()

    def get_output(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Compute error = state - reference.

        *reference* is [x1_ref, v1_ref, x2_ref, v2_ref].
        """
        return state - reference

    def get_constraints(self) -> Constraints:
        return Constraints(
            u_lb={"F1": -self.F_max, "F2": -self.F_max},
            u_ub={"F1": self.F_max, "F2": self.F_max},
            du_min={"F1": -self.dF_max, "F2": -self.dF_max},
            du_max={"F1": self.dF_max, "F2": self.dF_max},
            y_lb={},
            y_ub={},
        )

    def get_default_config_overrides(self) -> dict[str, Any]:
        return {
            "Q_diag": [20.0, 1.0, 20.0, 1.0],
            "R_diag": [0.1, 0.1],
            "T_data": 600,
            "noise_std_output": 0.005,
            "Tini": 4,
            "N": 20,
            "sim_duration": 30.0,
            "lambda_g": 10.0,
            "lambda_y": 1e4,
            "lambda_out": 0.0,
            "reg_norm_g": "L2",
            "reg_norm_sigma_y": "L1",
        }

    def get_scenarios(self) -> dict[str, Any]:
        return {
            "default": self._generate_step_sequence,
            "sinusoidal": self._generate_sinusoidal,
        }

    def get_data_collection_config(self) -> DataCollectionConfig:
        conditions = [
            {"x1_offset": 0.0, "x2_offset": 0.0},
            {"x1_offset": 0.5, "x2_offset": -0.5},
            {"x1_offset": -0.5, "x2_offset": 0.5},
            {"x1_offset": 0.3, "x2_offset": 0.3},
            {"x1_offset": -0.3, "x2_offset": -0.3},
        ]
        return DataCollectionConfig(
            initial_conditions=conditions,
            excitation_fn=self._generate_excitation,
            stabilizing_controller=self._stabilizing_controller,
            nominal_reference=self._nominal_reference,
        )

    def make_episode_initial_state(
        self, condition: dict[str, Any], rng: np.random.Generator,
    ) -> np.ndarray:
        x1_off = condition.get("x1_offset", 0.0)
        x2_off = condition.get("x2_offset", 0.0)
        v1 = rng.uniform(-0.2, 0.2)
        v2 = rng.uniform(-0.2, 0.2)
        return np.array([x1_off, v1, x2_off, v2])

    # ── Optional hooks ────────────────────────────────────────────────

    def get_initial_state_for_scenario(
        self, first_waypoint: np.ndarray,
    ) -> np.ndarray:
        return first_waypoint

    def get_position_from_state(self, state: np.ndarray) -> np.ndarray | None:
        return None  # not spatial

    def get_tuning_objective_key(self) -> str:
        return "rmse_e_x1"

    def compute_custom_metrics(self, results: dict) -> dict[str, float]:
        errors = np.asarray(results["y_history"])
        rmse_x1 = float(np.sqrt(np.mean(errors[:, 0] ** 2)))
        rmse_x2 = float(np.sqrt(np.mean(errors[:, 2] ** 2)))
        return {
            "rmse_position_combined": float(np.sqrt(
                np.mean(errors[:, 0] ** 2 + errors[:, 2] ** 2)
            )),
        }

    # ── Scenario generators ───────────────────────────────────────────

    def _generate_step_sequence(
        self, Tini: int, N: int, sim_steps: int, Ts: float,
    ) -> np.ndarray:
        """Alternating step setpoints for both masses."""
        total = Tini + sim_steps + N
        ref = np.zeros((total, 4))  # [x1, v1, x2, v2]

        # Mass 1: steps every ~5s with alternating sign
        step_len = int(5.0 / Ts)
        amplitudes_1 = [0.0, 1.0, -0.5, 1.5, -1.0, 0.5, -1.5, 1.0, 0.0]
        idx = 0
        for amp in amplitudes_1:
            end = min(idx + step_len, total)
            ref[idx:end, 0] = amp
            idx = end
            if idx >= total:
                break
        if idx < total:
            ref[idx:, 0] = amplitudes_1[-1]

        # Mass 2: steps offset by half a period
        half = step_len // 2
        amplitudes_2 = [0.0, -0.5, 1.0, -1.0, 0.5, 1.5, -0.5, 0.0]
        idx = half
        ref[:half, 2] = 0.0
        for amp in amplitudes_2:
            end = min(idx + step_len, total)
            ref[idx:end, 2] = amp
            idx = end
            if idx >= total:
                break
        if idx < total:
            ref[idx:, 2] = amplitudes_2[-1]

        # Reference velocities stay zero (we only track positions)
        return ref

    def _generate_sinusoidal(
        self, Tini: int, N: int, sim_steps: int, Ts: float,
    ) -> np.ndarray:
        """Sinusoidal position references at different frequencies."""
        total = Tini + sim_steps + N
        t = np.arange(total) * Ts
        ref = np.zeros((total, 4))
        ref[:, 0] = 1.0 * np.sin(2 * np.pi * 0.1 * t)
        ref[:, 2] = 0.8 * np.sin(2 * np.pi * 0.15 * t + np.pi / 3)
        return ref

    # ── Data collection helpers ───────────────────────────────────────

    def _generate_excitation(
        self, T_per: int, rng: np.random.Generator,
    ) -> np.ndarray:
        from sim.data_generation import generate_multisine, generate_prbs

        exc = np.zeros((T_per, 2))
        for ch in range(2):
            prbs = generate_prbs(T_per, self.F_max * 0.6, 3, rng)
            sine = generate_multisine(T_per, self.F_max * 0.6, 8, self._Ts, rng)
            exc[:, ch] = 0.5 * prbs + 0.5 * sine
        return exc

    def _stabilizing_controller(self, errors: np.ndarray) -> np.ndarray:
        """Simple proportional controller to keep errors bounded."""
        K_x = 1.5
        K_v = 0.8
        F1 = -K_x * errors[0] - K_v * errors[1]
        F2 = -K_x * errors[2] - K_v * errors[3]
        return np.array([F1, F2])

    def _nominal_reference(self, k: int, Ts: float) -> np.ndarray:
        """Nominal reference for data collection: all zeros (equilibrium)."""
        return np.zeros(4)
