"""What one run of the closed-loop piece leaves behind, and the panels and numbers it declares.

The arrays are the ones ``run_simulation`` returns, under names of their own: the time grid and
the outputs, inputs and reference over the warm-up and the loop (``time``), and one entry per
solve for the solver's own (``statuses``, ``cost``, ``sigma_y_norm``, ``lambda_g``,
``solve_time_s``). A value a solve left without one is NaN in its array, never dropped. Every
number is a reduction of an array on this record. Solves are counted by status, each status its
own number, so an inaccurate solve is never counted as an optimal one (D2); by the solver that
answered, and the solves that left no solution, where zeros were applied, are counted (D3),
all through ``sim.simulation.solve_counts``. Every axis carries
its unit, a channel's the plant's own (``input_units``, ``output_units``). The words follow the
developer's pattern (``docs/project/lab-words.md``). No chart declares a pooling statistic: the
closed loop holds no seed of its own, so its runs are never replicates of one run.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from record import Number, Provenance, Series
from record import Record as _Record

from .config import Config

_OUTPUTS, _INPUTS, _SOLVER, _POSITION = "Outputs", "Inputs", "Solver", "Position"


def _metric_words(name: str) -> str:
    """A plant metric's words from its code name: ``rmse_position`` reads "Position RMSE"."""
    if not name.startswith("rmse_"):
        return name[:1].upper() + name[1:].replace("_", " ")
    words = name[len("rmse_"):].replace("_", " ")
    return f"{words[:1].upper()}{words[1:]} RMSE"


@dataclasses.dataclass(frozen=True)
class Record(_Record):
    name: str
    config: Config
    provenance: Provenance | None
    constants: dict                 # the resolved DeePCConfig, the plant's step, the rank of H_L(u), the dataset run
    inputs: list                    # input channel names
    outputs: list                   # output channel names
    input_units: list               # the unit of each input, the plant's own
    output_units: list              # the unit of each output, the plant's own
    plant_metrics: list             # the names the plant's compute_custom_metrics returned
    plant_metric_units: list        # their units, the plant's own (metric_units)
    time: np.ndarray                # (T,) [s] on the config's step, warm-up included
    u: np.ndarray                   # (T, m) the controller's inputs, before the plant's own clipping
    y: np.ndarray                   # (T, p) outputs (tracking errors)
    reference: np.ndarray           # (T, ref_dim) the waypoints
    statuses: list                  # (K,) the solver's status per solve
    solvers: list                   # (K,) the solver that answered per solve (OSQP, or SCS after a SolverError)
    zero_input: list                # (K,) True where the solve left no solution and zeros were applied
    cost: np.ndarray                # (K,) objective per solve
    sigma_y_norm: np.ndarray        # (K,) 2-norm of the past-output slack per solve
    lambda_g: np.ndarray            # (K,) lambda_g used per solve
    solve_time_s: np.ndarray        # (K,) wall-clock time per solve
    position: np.ndarray | None     # (T+1, 2) plant position, one more than time: the start
    reference_position: np.ndarray | None   # (T, 2) the waypoints' positions
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
        t, step = self.time, np.arange(len(self.statuses))
        out = [Series(name=f"output-{c}".replace("_", "-"), x=t, x_label="Time [s]",
                      y_label=f"{c} [{unit}]", traces=((c, self.y[:, j]),), group=_OUTPUTS,
                      title=c, doc=f"{c} at each step, warm-up included")
               for j, (c, unit) in enumerate(zip(self.outputs, self.output_units))]
        out += [Series(name=f"input-{c}".replace("_", "-"), x=t, x_label="Time [s]",
                       y_label=f"{c} [{unit}]", traces=((c, self.u[:, j]),), group=_INPUTS,
                       title=c, doc=f"The controller's {c} at each step, warm-up included, before "
                                    f"the plant's own clipping")
                for j, (c, unit) in enumerate(zip(self.inputs, self.input_units))]
        out += [Series(name=name, x=step, x_label="Solve [-]", y_label=f"{title} [{unit}]",
                       traces=((title, values),), group=_SOLVER, title=title,
                       doc=f"{doc} at each solve")
                for name, title, unit, doc, values in (
                    ("solve-time", "Solve time", "s", "Wall-clock time of the solve",
                     self.solve_time_s),
                    ("cost", "J", "-", "Objective value J", self.cost),
                    ("sigma-y-norm", "‖σ_y‖₂", "-", "2-norm of the past-output slack σ_y",
                     self.sigma_y_norm),
                    ("lambda-g", "λ_g", "-", "λ_g the solve used", self.lambda_g))]
        if self.position is not None and self.reference_position is not None:
            n = len(self.reference_position)
            out += [Series(name=f"position-{axis}", x=np.arange(n), x_label="Step [-]",
                           y_label=f"{axis} [m]", title=f"Position {axis}",
                           traces=(("Plant", self.position[:n, j]),
                                   ("Reference", self.reference_position[:, j])),
                           group=_POSITION,
                           doc=f"The plant's {axis} after each step beside the waypoint of the "
                               f"same index, as the position RMSE compares them")
                    for j, axis in enumerate(("x", "y"))]
        return tuple(out)

    def number_specs(self) -> tuple[Number, ...]:
        out = [Number(f"rmse_{c}", f"{c} RMSE", unit, headline=(j == 0), group=_OUTPUTS,
                      doc=f"Root mean square of {c} over every step, warm-up included")
               for j, (c, unit) in enumerate(zip(self.outputs, self.output_units))]
        out += [Number(name, _metric_words(name), unit, group=_OUTPUTS,
                       doc="The plant's own metric (compute_custom_metrics)")
                for name, unit in zip(self.plant_metrics, self.plant_metric_units)]
        out += [Number(f"solves_{s}", f"Solves: {s.replace('_', ' ')}", group=_SOLVER,
                       doc=f"Solves whose status is exactly {s!r}")
                for s in self.status_counts()]
        out += [Number(f"solves_by_{s}", f"Solves by {s}", group=_SOLVER,
                       doc=f"Solves {s} answered" + (", after OSQP raised a SolverError"
                                                     if s == "SCS" else ""))
                for s in self.solver_counts()]
        out += [Number("zero_inputs_applied", "Zero inputs applied", group=_SOLVER,
                       doc="Solves that left no solution, after which the controller's zero "
                           "input was applied"),
                Number("solve_time_mean_s", "Mean solve time", "s", group=_SOLVER,
                       doc="Mean wall-clock time of a solve"),
                Number("solve_time_max_s", "Max solve time", "s", group=_SOLVER,
                       doc="Longest wall-clock time of a solve"),
                Number("sigma_y_norm_mean", "Mean ‖σ_y‖₂", group=_SOLVER,
                       doc="Mean 2-norm of the past-output slack σ_y"),
                Number("sigma_y_norm_max", "Max ‖σ_y‖₂", group=_SOLVER,
                       doc="Largest 2-norm of the past-output slack σ_y")]
        return tuple(out)
