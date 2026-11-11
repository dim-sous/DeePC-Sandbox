"""What one run of the closed-loop piece leaves behind, and the panels and numbers it declares.

The arrays are the ones ``run_simulation`` returns, under names of their own: the time grid and
the outputs, inputs and reference over the warm-up and the loop (``time``), and one entry per
solve for the solver's own (``statuses``, ``cost``, ``sigma_y_norm``, ``lambda_g``,
``solve_time_s``). A value a solve left without one is NaN in its array, never dropped. Every
number is a reduction of an array on this record. Solves are counted by status, each status its
own number, so an inaccurate solve is never counted as an optimal one (D2). Panel names, labels
and units are placeholders (RCWS #96): the plants declare no units, so an axis carries the
channel's code name only.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from record import Number, Provenance, Series
from record import Record as _Record

from .config import Config

_OUTPUTS, _INPUTS, _SOLVER, _POSITION = "Outputs", "Inputs", "Solver", "Position"


@dataclasses.dataclass(frozen=True)
class Record(_Record):
    name: str
    config: Config
    provenance: Provenance | None
    constants: dict                 # the resolved DeePCConfig, the plant's step and the config's, the dataset run
    inputs: list                    # input channel names
    outputs: list                   # output channel names
    plant_metrics: list             # the names the plant's compute_custom_metrics returned
    time: np.ndarray                # (T,) [s] on the config's step, warm-up included
    u: np.ndarray                   # (T, m) the controller's inputs, before the plant's own clipping
    y: np.ndarray                   # (T, p) outputs (tracking errors)
    reference: np.ndarray           # (T, ref_dim) the waypoints
    statuses: list                  # (K,) the solver's status per solve
    cost: np.ndarray                # (K,) objective per solve
    sigma_y_norm: np.ndarray        # (K,) 2-norm of the past-output slack per solve
    lambda_g: np.ndarray            # (K,) lambda_g used per solve
    solve_time_s: np.ndarray        # (K,) wall-clock time per solve
    position: np.ndarray | None     # (T+1, 2) plant position, one more than time: the start
    reference_position: np.ndarray | None   # (T, 2) the waypoints' positions
    numbers: dict

    component = "closed_loop"
    label = "closed_loop"
    produced_by = "uv run deepc run closed_loop [config.json]"

    def status_counts(self) -> dict[str, int]:
        """Solves per status, statuses in the order they first occur."""
        return {s: self.statuses.count(s) for s in dict.fromkeys(self.statuses)}

    def series(self) -> tuple[Series, ...]:
        t, step = self.time, np.arange(len(self.statuses))
        out = [Series(name=f"output-{c}".replace("_", "-"), x=t, x_label="time [s]", y_label=c,
                      traces=((c, self.y[:, j]),), group=_OUTPUTS,
                      doc=f"The {c} at each step, warm-up included (open words)")
               for j, c in enumerate(self.outputs)]
        out += [Series(name=f"input-{c}".replace("_", "-"), x=t, x_label="time [s]", y_label=c,
                       traces=((c, self.u[:, j]),), group=_INPUTS,
                       doc=f"The controller's {c} at each step, warm-up included, before the "
                           f"plant's own clipping (open words)")
                for j, c in enumerate(self.inputs)]
        out += [Series(name=name, x=step, x_label="solve", y_label=label, traces=((label, values),),
                       group=_SOLVER, doc=f"The {label} of each solve (open words)")
                for name, label, values in (("solve-time", "solve time [s]", self.solve_time_s),
                                            ("cost", "cost", self.cost),
                                            ("sigma-y-norm", "sigma_y norm", self.sigma_y_norm),
                                            ("lambda-g", "lambda_g", self.lambda_g))]
        if self.position is not None and self.reference_position is not None:
            n = len(self.reference_position)
            out += [Series(name=f"position-{axis}", x=np.arange(n), x_label="step",
                           y_label=f"{axis} [m]",
                           traces=(("plant", self.position[:n, j]),
                                   ("reference", self.reference_position[:, j])),
                           group=_POSITION,
                           doc=f"The plant's {axis} after each step beside the waypoint of the "
                               f"same index, as the position RMSE compares them; equal times only "
                               f"while Ts is the plant's own (D1) (open words)")
                    for j, axis in enumerate(("x", "y"))]
        return tuple(out)

    def number_specs(self) -> tuple[Number, ...]:
        out = [Number(f"rmse_{c}", f"{c} RMSE", headline=(j == 0), group=_OUTPUTS,
                      doc=f"Root mean square of the {c} over every step, warm-up included "
                          f"(open words)")
               for j, c in enumerate(self.outputs)]
        out += [Number(name, name, group=_POSITION if "position" in name or name in ("rmse_x", "rmse_y")
                       else _OUTPUTS, doc="The plant's own metric, compute_custom_metrics (open words)")
                for name in self.plant_metrics]
        out += [Number(f"solves_{s}", f"solves {s}", group=_SOLVER,
                       doc=f"Solves whose status is exactly {s!r} (open words)")
                for s in self.status_counts()]
        out += [Number("solve_time_mean_s", "mean solve time", "s", group=_SOLVER,
                       doc="Mean wall-clock time of a solve (open words)"),
                Number("solve_time_max_s", "longest solve time", "s", group=_SOLVER,
                       doc="Longest wall-clock time of a solve (open words)"),
                Number("sigma_y_norm_mean", "mean sigma_y norm", group=_SOLVER,
                       doc="Mean 2-norm of the past-output slack (open words)"),
                Number("sigma_y_norm_max", "largest sigma_y norm", group=_SOLVER,
                       doc="Largest 2-norm of the past-output slack (open words)")]
        return tuple(out)
