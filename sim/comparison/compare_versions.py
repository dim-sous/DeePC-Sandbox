"""Compare metrics across feature flag combinations.

Usage (from repo root):
    uv run python -m sim.comparison.compare_versions

Reads all ``results/<tag>/<tag>_metrics.json`` files and produces:
  - Console table
  - ``results/comparison.csv``
  - ``results/comparison.png``
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "results"

METRIC_KEYS = [
    "version",
    "rmse_x",
    "rmse_y",
    "rmse_v",
    "rmse_position",
    "max_position_error",
    "mean_abs_steering",
    "mean_abs_acceleration",
    "total_control_effort",
    "avg_solve_time_s",
    "max_solve_time_s",
    "total_solve_time_s",
    "optimal_solve_pct",
    "mean_sigma_y_norm",
    "max_sigma_y_norm",
]


def _fmt(val: float | str, width: int = 14) -> str:
    """Format a metric value for display."""
    if isinstance(val, str):
        return val.ljust(width)
    if np.isnan(val):
        return "N/A".center(width)
    if abs(val) < 0.001:
        return f"{val:.2e}".rjust(width)
    return f"{val:.4f}".rjust(width)


def _discover_metrics() -> list[dict]:
    """Find all metrics JSON files in results subdirectories."""
    all_metrics: list[dict] = []
    if not RESULTS_DIR.exists():
        return all_metrics
    for subdir in sorted(RESULTS_DIR.iterdir()):
        if subdir.is_dir():
            for mf in sorted(subdir.glob("*_metrics.json")):
                with open(mf) as f:
                    all_metrics.append(json.load(f))
    return all_metrics


def main() -> None:
    """Discover metrics files, print comparison, write CSV and plot."""
    all_metrics = _discover_metrics()
    if not all_metrics:
        print("No metrics files found in", RESULTS_DIR)
        sys.exit(1)

    # ---- Console table ----
    header = "  ".join(
        [f"{'Metric':<30}"] + [f"{m['version']:>16}" for m in all_metrics]
    )
    print("\n" + "=" * len(header))
    print("  COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for key in METRIC_KEYS:
        if key == "version":
            continue
        row = f"{key:<30}"
        for m in all_metrics:
            row += f"  {_fmt(m.get(key, float('nan')), 16)}"
        print(row)
    print("=" * len(header))

    # ---- CSV ----
    csv_path = RESULTS_DIR / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_KEYS)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: m.get(k, "") for k in METRIC_KEYS})
    print(f"\nCSV saved to {csv_path}")

    # ---- Bar chart comparison ----
    plot_keys = [
        ("rmse_position", "RMSE Position [m]"),
        ("rmse_v", "RMSE Velocity [m/s]"),
        ("total_control_effort", "Total Control Effort"),
        ("avg_solve_time_s", "Avg Solve Time [s]"),
        ("optimal_solve_pct", "Optimal Solves [%]"),
        ("mean_sigma_y_norm", "Mean Slack Norm"),
    ]

    versions = [m["version"] for m in all_metrics]
    n_metrics = len(plot_keys)
    n_versions = len(versions)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes_flat = axes.flatten()

    for i, (key, label) in enumerate(plot_keys):
        ax = axes_flat[i]
        vals = [m.get(key, float("nan")) for m in all_metrics]
        colours = plt.cm.tab10(np.linspace(0, 1, max(n_versions, 1)))
        bars = ax.bar(
            versions, vals, color=colours[:n_versions], edgecolor="k", linewidth=0.5
        )
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.4g}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    # Hide unused subplots
    for j in range(n_metrics, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved to {plot_path}")


if __name__ == "__main__":
    main()
