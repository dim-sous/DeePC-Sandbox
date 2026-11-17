"""The closed-loop piece's settings: the saved open-loop run it runs on (its data, and its plant:
the system, T_s, the actuator bounds, σ, σ_d and the integrator's tolerances are that run's, not
settings here), the loop's own initial state and seed, the reference typed as one row in t per
output with the loop's duration, and the controller's settings.

The defaults are today's coupled-masses posture, sized to the default system (n = 4, m = 2,
p = 4), as the open-loop piece's are (the developer's 2g, docs/project/design.md): T_ini, N, Q, R,
λ_g, λ_y and the two norms as ``c7f4569:tools/plants/coupled_masses.py`` set them, the
reference its ``sinusoidal`` scenario typed out, the duration its 30. The controller's maths is
not changed by this piece (the developer, 2026-11-17: "No math-based edit/decisions yet, only
framework/structure work"). The seed is the loop's own, by default not the data's 42 (the
developer, 2026-11-17). The names are the developer's: u, y, x, r, t; no units.
"""

from __future__ import annotations

import param

from benchlab import Settings

DATA, REFERENCE, CONTROLLER = "Data", "Reference", "Controller"
GROUPS: tuple[str, ...] = (DATA, REFERENCE, CONTROLLER)


class Config(Settings):
    """The closed-loop piece's settings; see the module."""

    open_loop_run = param.String(
        "results/open_loop/default.json", label="Open loop run",
        metadata={"group": DATA, "record_of": "open_loop"})
    # None is the open-loop run's own x_0, sized to whatever system it ran; the run records the
    # state it used
    x0 = param.List(
        None, allow_None=True, item_type=float, label="x_0",
        metadata={"group": DATA})
    seed = param.Integer(
        1, label="Seed",
        metadata={"group": DATA, "replicate": True})
    r = param.List(
        ["sin(2*pi*0.1*t)", "0", "0.8*sin(2*pi*0.15*t + pi/3)", "0"],
        item_type=str, label="r",
        metadata={"group": REFERENCE, "rows": {"label": "r{}", "add": "+ Reference"}})
    duration = param.Number(
        30.0, bounds=(0.0, None), inclusive_bounds=(False, True), label="Duration",
        metadata={"group": REFERENCE})
    Tini = param.Integer(
        4, bounds=(1, None), label="T_ini",
        metadata={"group": CONTROLLER})
    N = param.Integer(
        20, bounds=(1, None), label="N",
        metadata={"group": CONTROLLER})
    Q = param.List(
        [20.0, 1.0, 20.0, 1.0], item_type=float, label="Q",
        metadata={"group": CONTROLLER})
    R = param.List(
        [0.1, 0.1], item_type=float, label="R",
        metadata={"group": CONTROLLER})
    lambda_g = param.Number(
        10.0, bounds=(0.0, None), label="λ_g",
        metadata={"group": CONTROLLER})
    lambda_y = param.Number(
        1e4, bounds=(0.0, None), label="λ_y",
        metadata={"group": CONTROLLER})
    reg_norm_g = param.Selector(
        default="L2", objects=["L1", "L2"], label="Norm on g",
        metadata={"group": CONTROLLER})
    reg_norm_sigma_y = param.Selector(
        default="L1", objects=["L1", "L2"], label="Norm on σ_y",
        metadata={"group": CONTROLLER})
