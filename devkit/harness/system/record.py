"""What one run of the system piece leaves behind: the system as SymPy read its rows (θ kept as
symbols) and its sizes, and its analysis at the requested point (x*, u*) for the pair u_j -> y_i:
the linearisation (A, B, C), its poles, the pair's frequency response (Bode) and its step,
impulse and ramp responses, by python-control. Names as the developer names them (2026-11-11):
ω, |G|, ∠G, t, δy; no units (the phase is in degrees, the hovers say so).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from record import Number, Provenance, Series
from record import Record as _Record

from .config import Config

_SIZES, _POLES, _PAIR = "Sizes", "Poles", "Pair"
_FREQUENCY, _TIME = "Frequency", "Time"


@dataclasses.dataclass(frozen=True)
class Record(_Record):
    name: str
    config: Config
    provenance: Provenance | None
    constants: dict          # rows as read; linearisation, poles, ranges; why a number is left out
    omega: np.ndarray        # (F,) the frequencies python-control chose
    magnitude: np.ndarray    # (F,) |G(jω)| of the pair
    phase_deg: np.ndarray    # (F,) ∠G(jω) of the pair, unwrapped, in degrees
    t: np.ndarray            # (K,) the times python-control chose for the step response
    step: np.ndarray         # (K,) δy_i after a unit step in u_j
    impulse: np.ndarray      # (K,) δy_i after a unit impulse in u_j
    ramp: np.ndarray         # (K,) δy_i under u_j = t
    numbers: dict

    component = "system"
    label = "System"
    produced_by = "uv run deepc run system [config.json]"

    def series(self) -> tuple[Series, ...]:
        i, j = self.config.output, self.config.input
        pair = f"u{j} → y{i} at (x*, u*)"
        out = [Series(name="bode-magnitude", x=self.omega, x_label="ω", y_label="|G|",
                      traces=((f"|G| of u{j} → y{i}", self.magnitude),), group=_FREQUENCY,
                      title="Bode magnitude", x_log=True, y_log=True,
                      doc=f"The magnitude of the frequency response of {pair}"),
               Series(name="bode-phase", x=self.omega, x_label="ω", y_label="∠G",
                      traces=((f"∠G of u{j} → y{i}", self.phase_deg),), group=_FREQUENCY,
                      title="Bode phase", x_log=True,
                      doc=f"The phase of the frequency response of {pair}, in degrees")]
        out += [Series(name=kind, x=self.t, x_label="t", y_label=f"δy{i}",
                       traces=((f"δy{i}", values),), group=_TIME, title=words,
                       doc=f"y{i}'s deviation from h(x*) {how}, linearised at (x*, u*)")
                for kind, words, values, how in (
                    ("step", "Step response", self.step, f"after a unit step in u{j}"),
                    ("impulse", "Impulse response", self.impulse, f"after a unit impulse in u{j}"),
                    ("ramp", "Ramp response", self.ramp, f"under u{j} - u{j}* = t"))]
        return tuple(out)

    def number_specs(self) -> tuple[Number, ...]:
        sizes = (("n", "The number of states, x1…xn: one per state row"),
                 ("m", "The number of inputs, u1…um"),
                 ("p", "The number of outputs, y1…yp: one per output row"),
                 ("q", "The number of parameters, theta1…thetaq"),
                 ("r", "The number of disturbance inputs, d1…dr: where the unmodelled "
                       "disturbance enters"))
        out = [Number(name, name, group=_SIZES, doc=doc) for name, doc in sizes]
        analysis = (
            ("unstable_poles", "Unstable poles", _POLES,
             "Eigenvalues of A with a positive real part, at (x*, u*)"),
            ("max_pole_re", "Max pole Re", _POLES, "The largest real part of A's eigenvalues"),
            ("min_damping", "Min damping ratio", _POLES,
             "The smallest damping ratio of A's eigenvalues (python-control's damp); a pole at "
             "the origin has none"),
            ("residual", "Equilibrium residual", _POLES,
             "|f(x*, u*, 0)|: 0 when (x*, u*) is an equilibrium"),
            ("dc_gain", "DC gain", _PAIR, "The pair's steady-state gain G(0)"),
            ("gain_margin", "Gain margin", _PAIR,
             "The pair's gain margin, as python-control's stability_margins computes it"),
            ("phase_margin", "Phase margin", _PAIR,
             "The pair's phase margin in degrees, as python-control's stability_margins "
             "computes it"))
        out += [Number(name, label, group=group, headline=(name == "unstable_poles"), doc=doc)
                for name, label, group, doc in analysis if name in self.numbers]
        return tuple(out)
