"""What one run of the closed-loop piece leaves behind: at every sample, the inputs the plant
applied, the outputs it measured (noise included) and the same outputs before the noise, the
reference, the true states and the disturbances; at every solve, the solver's status, the solver
that answered, the objective, ‖σ_y‖₂ and the solve's wall-clock time. One chart per channel
against t, the reference on its output's chart and the actuator bounds on their input's, and the
solver's charts against the time of each solve; named as the developer names them (2026-11-11):
u1…, y1…, x1…, r1…, d1…, t, no units. The true states are drawn for the developer; the
controller never reads them.

Every number is a reduction of an array on this record. The RMSE of each output is taken on
h(x_k) - r_k, the plant's own output before the noise, not on the measured y (CLAUDE.md: the
plant's state generates the scores). Solves are counted by status, each status its own number
(D2), and by the solver that answered (D3).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from benchlab import Number, Provenance, Series
from benchlab import Record as _Record

from .config import Config

_OUTPUTS, _INPUTS, _STATES, _DISTURBANCES, _SOLVER = (
    "Outputs", "Inputs", "States", "Disturbances", "Solver")


@dataclasses.dataclass(frozen=True)
class Record(_Record):
    name: str
    config: Config
    provenance: Provenance | None
    constants: dict          # the open-loop run and its plant; r as read; the controller as built
    t: np.ndarray            # (K,) the sample times k T_s, warm-up included
    u: np.ndarray            # (K, m) the inputs the plant applied: the controller's, then clipped
    y: np.ndarray            # (K, p) the outputs measured, h(x_k) plus noise: what DeePC reads
    y_true: np.ndarray       # (K, p) h(x_k), the outputs before the noise
    r: np.ndarray            # (K, p) the reference r(t_k)
    x: np.ndarray            # (K, n) the true states x_k
    d: np.ndarray            # (K, r) the disturbances d_k, held over each step
    statuses: list           # (S,) the solver's status per solve, S = K - T_ini
    solvers: list            # (S,) the solver that answered per solve (OSQP, or SCS after a SolverError)
    cost: np.ndarray         # (S,) the objective per solve
    sigma_y_norm: np.ndarray  # (S,) ‖σ_y‖₂ per solve
    solve_time: np.ndarray   # (S,) wall-clock time per solve
    numbers: dict

    component = "closed_loop"
    label = "Closed loop"
    produced_by = "uv run deepc run closed_loop [config.json]"

    def status_counts(self) -> dict[str, int]:
        """Solves per status, statuses in the order they first occur."""
        return {s: self.statuses.count(s) for s in dict.fromkeys(self.statuses)}

    def solver_counts(self) -> dict[str, int]:
        """Solves per solver that answered, solvers in the order they first occur."""
        return {s: self.solvers.count(s) for s in dict.fromkeys(self.solvers)}

    def series(self) -> tuple[Series, ...]:
        def bounds(j: int) -> tuple[tuple[str, float], ...]:
            lows, highs = self.constants["u_min"], self.constants["u_max"]
            return tuple((f"{side} bound of u{j + 1}", float(v))
                         for side, v in (("Lower", lows[j]), ("Upper", highs[j]))
                         if np.isfinite(v))
        out = [Series(name=f"y{i + 1}", x=self.t, x_label="t", y_label=f"y{i + 1}",
                      traces=((f"y{i + 1}", self.y[:, i]), (f"r{i + 1}", self.r[:, i])),
                      group=_OUTPUTS, title=f"y{i + 1}")
               for i in range(self.y.shape[1])]
        out += [Series(name=f"u{j + 1}", x=self.t, x_label="t", y_label=f"u{j + 1}",
                       traces=((f"u{j + 1}", self.u[:, j]),), refs=bounds(j), group=_INPUTS,
                       title=f"u{j + 1}")
                for j in range(self.u.shape[1])]
        out += [Series(name=f"x{i + 1}", x=self.t, x_label="t", y_label=f"x{i + 1}",
                       traces=((f"x{i + 1}", self.x[:, i]),), group=_STATES, title=f"x{i + 1}")
                for i in range(self.x.shape[1])]
        out += [Series(name=f"d{k + 1}", x=self.t, x_label="t", y_label=f"d{k + 1}",
                       traces=((f"d{k + 1}", self.d[:, k]),), group=_DISTURBANCES,
                       title=f"d{k + 1}")
                for k in range(self.d.shape[1])]
        solved = self.t[len(self.t) - len(self.statuses):]      # the samples a solve chose u at
        out += [Series(name=name, x=solved, x_label="t", y_label=title,
                       traces=((title, values),), group=_SOLVER, title=title)
                for name, title, values in (("solve-time", "Solve time", self.solve_time),
                                            ("cost", "J", self.cost),
                                            ("sigma-y-norm", "‖σ_y‖₂", self.sigma_y_norm))]
        return tuple(out)

    def number_specs(self) -> tuple[Number, ...]:
        out = [Number(f"y{i + 1}_rmse", f"y{i + 1} RMSE", headline=(i == 0), group=_OUTPUTS)
               for i in range(self.y.shape[1])]
        out += [Number("clipped", "Clipped inputs", group=_INPUTS)]
        out += [Number(f"solves_{s}", f"Solves: {s.replace('_', ' ')}", group=_SOLVER)
                for s in self.status_counts()]
        out += [Number(f"solves_by_{s}", f"Solves by {s}", group=_SOLVER)
                for s in self.solver_counts()]
        out += [Number("solve_time_mean", "Mean solve time", group=_SOLVER),
                Number("solve_time_std", "Solve time std", group=_SOLVER),
                Number("solve_time_max", "Max solve time", group=_SOLVER),
                Number("sigma_y_norm_mean", "Mean ‖σ_y‖₂", group=_SOLVER),
                Number("sigma_y_norm_max", "Max ‖σ_y‖₂", group=_SOLVER)]
        return tuple(out)
