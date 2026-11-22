"""The open-loop piece's settings: the saved system run it simulates, its initial state, the
actuator bounds and the excitation range of each input, the sampling, the measurement noise, the
disturbances' sizes and the integrator's tolerances (the developer, 2026-11-11: "Bounds, initial
conditions, sampling rates etc live in the open loop tab"). The defaults are the values today's
two-masses runs use, sized to the system piece's default (n = 4, m = 2, r = 2); σ_d = [0.1, 0.1]
is the developer's preview (2026-11-11). The names are the developer's: u, y, x, d; no units.
"""

from __future__ import annotations

import param

from runlab import Settings
from system import ATOL, RTOL

SYSTEM, INPUTS, SIMULATION, INTEGRATOR = "System", "Inputs", "Simulation", "Integrator"
GROUPS: tuple[str, ...] = (SYSTEM, INPUTS, SIMULATION, INTEGRATOR)


class Config(Settings):
    """The open-loop piece's settings; see the module."""

    system_run = param.String(
        "results/system/default.json", label="System run",
        metadata={"group": SYSTEM, "record_of": "system"})
    x0 = param.List(
        [0.5, 0.0, -0.5, 0.0], item_type=float, label="x_0",
        metadata={"group": SYSTEM})
    u_min = param.List(
        [-5.0, -5.0], item_type=float, label="u_min",
        metadata={"group": INPUTS})
    u_max = param.List(
        [5.0, 5.0], item_type=float, label="u_max",
        metadata={"group": INPUTS})
    u_exc_min = param.List(
        [-3.0, -3.0], item_type=float, label="u_exc_min",
        metadata={"group": INPUTS})
    u_exc_max = param.List(
        [3.0, 3.0], item_type=float, label="u_exc_max",
        metadata={"group": INPUTS})
    Ts = param.Number(
        0.05, bounds=(0.0, None), inclusive_bounds=(False, True), label="T_s",
        metadata={"group": SIMULATION})
    T = param.Integer(
        600, bounds=(1, None), label="T",
        metadata={"group": SIMULATION})
    sigma = param.Number(
        0.005, bounds=(0.0, None), label="σ",
        metadata={"group": SIMULATION})
    sigma_d = param.List(
        [0.1, 0.1], item_type=float, label="σ_d",
        metadata={"group": SIMULATION})
    seed = param.Integer(
        42, label="Seed",
        metadata={"group": SIMULATION, "replicate": True})
    rtol = param.Number(
        RTOL, bounds=(0.0, None), inclusive_bounds=(False, True), label="rtol",
        metadata={"group": INTEGRATOR, "folded": True})
    atol = param.Number(
        ATOL, bounds=(0.0, None), inclusive_bounds=(False, True), label="atol",
        metadata={"group": INTEGRATOR, "folded": True})
