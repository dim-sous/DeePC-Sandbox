"""What one run of the system piece leaves behind: the system as SymPy read its rows (θ kept as
symbols) and its sizes, and its analysis at the requested point (x*, u*) for the pair u_j -> y_i:
the linearisation (A, B, C), its poles, the pair's frequency response (Bode) and its step,
impulse and ramp responses, by python-control. Names as the developer names them (2026-11-11):
ω, |G|, ∠G, t, δy; no units (the phase is in degrees, by the Bode plot's convention). The poles
are also drawn, on the complex plane: a figure the record draws itself, since a chart is lines
against an abscissa (RunLab's ``Figure``).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from runlab import Figure, Number, Provenance, Series
from runlab import Record as _Record

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
        out = [Series(name="bode-magnitude", x=self.omega, x_label="ω", y_label="|G|",
                      traces=((f"|G| of u{j} → y{i}", self.magnitude),), group=_FREQUENCY,
                      title="Bode magnitude", x_log=True, y_log=True),
               Series(name="bode-phase", x=self.omega, x_label="ω", y_label="∠G",
                      traces=((f"∠G of u{j} → y{i}", self.phase_deg),), group=_FREQUENCY,
                      title="Bode phase", x_log=True)]
        out += [Series(name=kind, x=self.t, x_label="t", y_label=f"δy{i}",
                       traces=((f"δy{i}", values),), group=_TIME, title=words)
                for kind, words, values in (("step", "Step response", self.step),
                                            ("impulse", "Impulse response", self.impulse),
                                            ("ramp", "Ramp response", self.ramp))]
        return tuple(out)

    def figures(self) -> tuple[Figure, ...]:
        """The poles of A on the complex plane, a cross each, both axes on one scale."""
        return (Figure(name="pole-map", group=_POLES, title="Pole map", figure={
            "data": [{"type": "scatter", "mode": "markers", "x": self.constants["poles_re"],
                      "y": self.constants["poles_im"], "marker": {"symbol": "x", "size": 11},
                      "showlegend": False}],
            "layout": {"xaxis": {"title": {"text": "Re"}, "zeroline": True},
                       "yaxis": {"title": {"text": "Im"}, "zeroline": True,
                                 "scaleanchor": "x"}}}),)

    def number_specs(self) -> tuple[Number, ...]:
        out = [Number(name, name, group=_SIZES) for name in ("n", "m", "p", "q", "r")]
        analysis = (("unstable_poles", "Unstable poles", _POLES),
                    ("max_pole_re", "Max pole Re", _POLES),
                    ("min_damping", "Min damping ratio", _POLES),
                    ("residual", "Equilibrium residual", _POLES),
                    ("dc_gain", "DC gain", _PAIR),
                    ("gain_margin", "Gain margin", _PAIR),
                    ("phase_margin", "Phase margin", _PAIR))
        out += [Number(name, label, group=group, headline=(name == "unstable_poles"))
                for name, label, group in analysis if name in self.numbers]
        return tuple(out)
