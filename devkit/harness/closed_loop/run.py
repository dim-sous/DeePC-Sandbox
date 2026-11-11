"""``run(config) -> Record``: DeePC in closed loop on a saved dataset run, as ``run.py`` runs it.
No printing (``run_simulation``'s progress lines are taken), no saving, no drawing.

The dataset run is read through the dataset piece (``harness.dataset.dataset_of``: run again and
checked); its plant is built fresh at the dataset run's step (``harness.dataset.plant_of``), the
``DeePCConfig`` is built from it as ``run.py`` builds one
(``sim.config.build_deepc_config``, a setting left at ``None`` taking the code's own default),
and ``run_simulation`` runs with the dataset's inputs and outputs.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import time as _time

import numpy as np

from harness.dataset import dataset_of, plant_of
from record import stamp
from sim.bounds import input_bounds_of
from sim.config import build_deepc_config
from sim.scenarios import get_reference
from sim.simulation import FeatureFlags, run_simulation, solve_counts

from .config import DEEPC_SETTINGS, FEATURE_SETTINGS, Config
from .record import Record


def _floats(values: list) -> np.ndarray:
    """A per-solve list as floats, a solve that left no value as NaN."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _tightened(plant, deepc_cfg, cfg: Config) -> bool:
    """Whether the noise-adaptive tightening enters any constraint: only output bounds take it,
    and the controller holds them only with ``lambda_out > 0``, a finite bound and the bounds on
    (``deepc.controller``, the output-constraint block)."""
    bounds = plant.get_constraints()
    finite = any(np.isfinite(v) for v in (*bounds.y_lb.values(), *bounds.y_ub.values()))
    return deepc_cfg.lambda_out > 0 and finite and not cfg.no_constraints


def run(cfg: Config) -> Record:
    started = _time.perf_counter()
    data = dataset_of(cfg.dataset_run)
    plant = plant_of(data.config)
    if cfg.scenario not in plant.get_scenarios():
        raise ValueError(f"the plant {data.config.plant!r} of {cfg.dataset_run} offers the "
                         f"scenarios {sorted(plant.get_scenarios())}, not {cfg.scenario!r}")
    settings = {k: getattr(cfg, k) for k in DEEPC_SETTINGS}
    for k in ("Q_diag", "R_diag"):
        settings[k] = None if settings[k] is None else list(settings[k])
    deepc_cfg = build_deepc_config(plant, T_data=data.constants["T_data"],
                                   noise_std_output=data.constants["noise_std_output"], **settings)
    if cfg.constraint_tightening_factor is not None and not _tightened(plant, deepc_cfg, cfg):
        raise ValueError(f"constraint_tightening_factor={cfg.constraint_tightening_factor!r} is "
                         f"set, but the tightening enters only output bounds, which this run does "
                         f"not have (lambda_out={deepc_cfg.lambda_out}, the plant's finite output "
                         f"bounds, no_constraints={cfg.no_constraints}; D11); the run would "
                         f"ignore it")
    path = get_reference(plant, cfg.scenario, deepc_cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_simulation(plant, deepc_cfg, FeatureFlags(**{k: getattr(cfg, k) for k in FEATURE_SETTINGS}),
                             data.u, data.y, path)

    y = np.asarray(res["y_history"])
    sigma = _floats(res["sigma_norms"])
    solve = np.asarray(res["solve_times"], dtype=float)
    plant_metrics = plant.compute_custom_metrics(res)
    unitless = sorted(set(plant_metrics) - set(plant.metric_units))
    if unitless:
        raise ValueError(f"the plant {data.config.plant!r} returns the metrics {unitless} with no "
                         f"unit (metric_units); a number is recorded with its unit")
    numbers = {f"rmse_{c}": float(np.sqrt(np.mean(y[:, j] ** 2)))
               for j, c in enumerate(plant.output_names)}
    numbers.update({k: float(v) for k, v in plant_metrics.items()})
    statuses = [str(s) for s in res["statuses"]]
    numbers.update({k: float(v) for k, v in solve_counts(res).items()})
    numbers["solve_time_mean_s"] = float(np.mean(solve))
    numbers["solve_time_max_s"] = float(np.max(solve))
    if np.all(np.isfinite(sigma)):          # a solve with no slack leaves these numbers out
        numbers["sigma_y_norm_mean"] = float(np.mean(sigma))
        numbers["sigma_y_norm_max"] = float(np.max(sigma))

    return Record(
        name="default", config=cfg, provenance=stamp(_time.perf_counter() - started),
        constants={"plant": data.config.plant, "plant_Ts_s": plant.Ts, "sim_steps": deepc_cfg.sim_steps,
                   "excitation_rank": res["excitation_rank"],
                   "excitation_rank_required": res["excitation_rank_required"],
                   "deepc_config": dataclasses.asdict(deepc_cfg),
                   "dataset_run": {"path": cfg.dataset_run, "name": data.name,
                                   "date": data.provenance.date if data.provenance else "",
                                   "git_sha": data.provenance.git_sha if data.provenance else ""}},
        inputs=list(plant.input_names), outputs=list(plant.output_names),
        input_units=list(plant.input_units), input_bounds=input_bounds_of(plant),
        output_units=list(plant.output_units),
        plant_metrics=list(plant_metrics),
        plant_metric_units=[plant.metric_units[k] for k in plant_metrics],
        time=np.asarray(res["times"], dtype=float), u=np.asarray(res["u_history"], dtype=float), y=y,
        reference=np.asarray(res["ref_path"], dtype=float), statuses=statuses,
        solvers=[str(s) for s in res["solvers"]], zero_input=[bool(z) for z in res["zero_input"]],
        cost=_floats(res["costs"]), sigma_y_norm=sigma, lambda_g=_floats(res["lambda_g_history"]),
        solve_time_s=solve,
        numbers=numbers)
