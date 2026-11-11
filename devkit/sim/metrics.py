"""The metrics ``tune.py`` scores a trial by, moved unchanged from the retired ``run.py`` (PR #10):
one dictionary per closed loop, read off what ``run_simulation`` returns."""

from __future__ import annotations

import numpy as np

from plants.base import PlantBase
from sim.simulation import solve_counts


def compute_metrics(
    results: dict, plant: PlantBase, tag: str,
) -> dict[str, float | str]:
    """Compute performance metrics from simulation results."""
    metrics: dict[str, float | str] = {"version": tag}

    # Per-channel output RMSE
    errors = np.asarray(results["y_history"])
    for i, name in enumerate(plant.output_names):
        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean(errors[:, i] ** 2)))

    # Plant-specific custom metrics (e.g. position RMSE for bicycle)
    metrics.update(plant.compute_custom_metrics(results))

    # Solver stats
    solve_times = results.get("solve_times")
    if solve_times is not None and len(solve_times) > 0:
        metrics["avg_solve_time_s"] = float(np.mean(solve_times))
        metrics["max_solve_time_s"] = float(np.max(solve_times))
        metrics["total_solve_time_s"] = float(np.sum(solve_times))
    else:
        metrics["avg_solve_time_s"] = float("nan")
        metrics["max_solve_time_s"] = float("nan")
        metrics["total_solve_time_s"] = float("nan")

    statuses = results.get("statuses", [])
    if statuses:
        counts = solve_counts(results)          # each status by its own name (D2), solvers, D3
        metrics["optimal_solve_pct"] = float(100.0 * counts.get("solves_optimal", 0) / len(statuses))
        metrics.update({k: float(v) for k, v in counts.items()})
        metrics["excitation_rank"] = float(results["excitation_rank"])            # D3
        metrics["excitation_rank_required"] = float(results["excitation_rank_required"])
    else:
        metrics["optimal_solve_pct"] = float("nan")

    sigma_norms = results.get("sigma_norms")
    if sigma_norms is not None:
        norms = [s for s in sigma_norms if s is not None]
        if norms:
            metrics["mean_sigma_y_norm"] = float(np.mean(norms))
            metrics["max_sigma_y_norm"] = float(np.max(norms))
        else:
            metrics["mean_sigma_y_norm"] = float("nan")
            metrics["max_sigma_y_norm"] = float("nan")

    return metrics
