"""The system piece's settings: the system as typed rows, x' = f(x, u, d, θ) one row per state
(d the disturbance inputs, where the unmodelled disturbance enters) and y = h(x, θ) one row per
output, and the values of θ. The initial state, the input bounds and the
sampling time are the open-loop simulation's (the developer, 2026-11-11). The names are the
developer's: inputs u, outputs y, states x, parameters θ; no units.
"""

from __future__ import annotations

import param

from record.config import Settings

# words, not the symbols: the form writes its group headings in capitals, which would turn x
# into X; the rows themselves are labelled x1', y1, θ1
X, Y, THETA, ANALYSIS = "States", "Outputs", "Parameters", "Analysis"
GROUPS: tuple[str, ...] = (X, Y, THETA, ANALYSIS)


class Config(Settings):
    """The system piece's settings; see the module."""

    # The coupled masses of tools/plants/coupled_masses.py at its defaults, m1 = m2 = 1 and
    # θ = [k1 = k2, k12, c1 = c2, c12] = [2, 1, 0.3, 0.1], typed out (the developer's pick,
    # 2026-11-11). Copied, not read: that plant class is deleted by the rewrite's third branch.
    # The disturbances d1, d2 act as forces on the two masses, beside u1, u2 (the developer's
    # preview, 2026-11-11).
    f = param.List(
        ["x2", "-(theta1 + theta2)*x1 - (theta3 + theta4)*x2 + theta2*x3 + theta4*x4 + u1 + d1",
         "x4", "theta2*x1 + theta4*x2 - (theta1 + theta2)*x3 - (theta3 + theta4)*x4 + u2 + d2"],
        item_type=str, label="x'",
        doc="One row per state: x_i' as an expression in x1…, u1…, the disturbances d1…, "
            "theta1… and sin cos tan exp log sqrt tanh abs pi",
        metadata={"group": X, "rows": {"label": "x{}'", "add": "+ State"}})
    h = param.List(
        ["x1", "x2", "x3", "x4"], item_type=str, label="y",
        doc="One row per output: y_j as an expression in x1…, theta1… and the same functions",
        metadata={"group": Y, "rows": {"label": "y{}", "add": "+ Output"}})
    theta = param.List(
        [2.0, 1.0, 0.3, 0.1], item_type=float, label="θ",
        doc="One row per parameter: the value of theta_k, which the rows name as theta1, theta2, …",
        metadata={"group": THETA, "rows": {"label": "θ{}", "add": "+ Parameter"}})
    # The analysis request: the point to linearise at and the input-output pair (the developer,
    # 2026-11-11: "the linearization is not fixed but defined by the user during plot request").
    # None is the origin, sized to whatever system the rows define, so a new system runs without
    # them typed; the run records the point it used.
    x_star = param.List(
        None, allow_None=True, item_type=float, label="x*",
        doc="The state to linearise at, one value per state; none: the origin",
        metadata={"group": ANALYSIS})
    u_star = param.List(
        None, allow_None=True, item_type=float, label="u*",
        doc="The input to linearise at, one value per input; none: zero",
        metadata={"group": ANALYSIS})
    input = param.Integer(
        1, bounds=(1, None), label="Input",
        doc="The input u_j of the analysed pair, by its index j",
        metadata={"group": ANALYSIS})
    output = param.Integer(
        1, bounds=(1, None), label="Output",
        doc="The output y_i of the analysed pair, by its index i",
        metadata={"group": ANALYSIS})
