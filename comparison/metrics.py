"""Metrics computation for DeePC platform version comparison.

Computes all mandatory performance metrics from a simulation results dict.
Shared across all version folders.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np


def compute_all_metrics(
    results: dict[str, Any],
    version_tag: str,
) -> dict[str, float | str]:
    """Compute all comparison metrics from simulation results.

    Parameters
    ----------
    results : dict
        Output from ``run_deepc_simulation()``.
    version_tag : str
        Version identifier (e.g. ``"v1_baseline"``).

    Returns
    -------
    dict
        Flat dictionary with all metric values.
    """
    metrics: dict[str, float | str] = {"version": version_tag}

    y_hist = np.asarray(results["y_history"])
    y_ref = np.asarray(results["y_ref_history"])
    u_hist = np.asarray(results["u_history"])

    # Align lengths
    n = min(len(y_hist), len(y_ref), len(u_hist))
    y_hist = y_hist[:n]
    y_ref = y_ref[:n]
    u_hist = u_hist[:n]

    # ------------------------------------------------------------------
    #  Tracking performance
    # ------------------------------------------------------------------
    errors = y_hist - y_ref
    metrics["rmse_x"] = float(np.sqrt(np.mean(errors[:, 0] ** 2)))
    metrics["rmse_y"] = float(np.sqrt(np.mean(errors[:, 1] ** 2)))
    metrics["rmse_v"] = float(np.sqrt(np.mean(errors[:, 2] ** 2)))
    metrics["rmse_position"] = float(
        np.sqrt(np.mean(errors[:, 0] ** 2 + errors[:, 1] ** 2))
    )
    metrics["max_position_error"] = float(
        np.max(np.sqrt(errors[:, 0] ** 2 + errors[:, 1] ** 2))
    )

    # ------------------------------------------------------------------
    #  Control effort
    # ------------------------------------------------------------------
    metrics["mean_abs_steering"] = float(np.mean(np.abs(u_hist[:, 0])))
    metrics["mean_abs_acceleration"] = float(np.mean(np.abs(u_hist[:, 1])))
    metrics["total_control_effort"] = float(np.sum(u_hist ** 2))

    # ------------------------------------------------------------------
    #  Solver performance
    # ------------------------------------------------------------------
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
        optimal_count = sum(1 for s in statuses if "optimal" in s)
        metrics["optimal_solve_pct"] = float(
            100.0 * optimal_count / len(statuses)
        )
    else:
        metrics["optimal_solve_pct"] = float("nan")

    # ------------------------------------------------------------------
    #  Constraint violations
    # ------------------------------------------------------------------
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


def save_metrics(metrics: dict, path: str | pathlib.Path) -> None:
    """Save metrics dict to JSON."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")
