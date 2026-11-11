"""What one run of the open-loop piece leaves behind: the inputs the plant applied, the outputs it
measured (noise included) and its true states, at every sample; one chart per channel against t,
named as the developer names them (2026-11-11): u1…, y1…, x1…, no units. The true states are drawn
for the developer; the controller never reads them.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from record import Number, Provenance, Series
from record import Record as _Record

from .config import Config

_INPUTS, _OUTPUTS, _STATES, _DISTURBANCES = "Inputs", "Outputs", "States", "Disturbances"
_COST = "Cost"


@dataclasses.dataclass(frozen=True)
class Record(_Record):
    name: str
    config: Config
    provenance: Provenance | None
    constants: dict          # the system run (path, rows, θ, sizes); clipped samples per input
    t: np.ndarray            # (T,) the sample times k T_s
    u: np.ndarray            # (T, m) the inputs the plant applied: drawn, then clipped
    y: np.ndarray            # (T, p) the outputs measured, h(x_k) plus noise
    x: np.ndarray            # (T, n) the true states x_k
    d: np.ndarray            # (T, r) the disturbances d_k, held over each step
    numbers: dict

    component = "open_loop"
    label = "Open loop"
    produced_by = "uv run deepc run open_loop [config.json]"

    def series(self) -> tuple[Series, ...]:
        def bounds(j: int) -> tuple[tuple[str, float], ...]:
            return tuple((f"{side} bound of u{j + 1}", float(v))
                         for side, v in (("Lower", self.config.u_min[j]),
                                         ("Upper", self.config.u_max[j])) if np.isfinite(v))
        out = [Series(name=f"u{j + 1}", x=self.t, x_label="t", y_label=f"u{j + 1}",
                      traces=((f"u{j + 1}", self.u[:, j]),), refs=bounds(j), group=_INPUTS,
                      title=f"u{j + 1}",
                      doc=f"The u{j + 1} the plant applied at each sample, held over T_s; its "
                          f"actuator bounds")
               for j in range(self.u.shape[1])]
        out += [Series(name=f"y{j + 1}", x=self.t, x_label="t", y_label=f"y{j + 1}",
                       traces=((f"y{j + 1}", self.y[:, j]),), group=_OUTPUTS, title=f"y{j + 1}",
                       doc=f"y{j + 1} measured at each sample, noise included")
                for j in range(self.y.shape[1])]
        out += [Series(name=f"x{i + 1}", x=self.t, x_label="t", y_label=f"x{i + 1}",
                       traces=((f"x{i + 1}", self.x[:, i]),), group=_STATES, title=f"x{i + 1}",
                       doc=f"The true x{i + 1} at each sample, which the controller never reads")
                for i in range(self.x.shape[1])]
        out += [Series(name=f"d{k + 1}", x=self.t, x_label="t", y_label=f"d{k + 1}",
                       traces=((f"d{k + 1}", self.d[:, k]),), group=_DISTURBANCES,
                       title=f"d{k + 1}",
                       doc=f"The disturbance d{k + 1} held over each step, which the controller "
                           f"never reads")
                for k in range(self.d.shape[1])]
        return tuple(out)

    def number_specs(self) -> tuple[Number, ...]:
        out = [Number("pe_order", "PE order of u", group=_INPUTS, headline=True,
                      doc="The largest L at which the input's Hankel matrix H_L(u) has full row "
                          "rank L·m: persistency of excitation of order L"),
               Number("clipped", "Clipped inputs", group=_INPUTS,
                      doc="Input samples the actuator bounds changed, over all inputs")]
        out += [Number(f"y{j + 1}_snr", f"y{j + 1} SNR", group=_OUTPUTS,
                       doc=f"Variance of the noise-free y{j + 1}, h(x), over the measurement "
                           f"noise's variance σ²")
                for j in range(self.y.shape[1]) if f"y{j + 1}_snr" in self.numbers]
        out += [
                Number("step_time_mean", "Mean step time", group=_COST,
                       doc="Mean wall-clock seconds of one sampling step: the clip and the "
                           "integration"),
                Number("step_time_std", "Step time std", group=_COST,
                       doc="Standard deviation of the wall-clock seconds of one sampling step"),
                Number("step_time_max", "Max step time", group=_COST,
                       doc="Longest wall-clock seconds of one sampling step")]
        return tuple(out)
