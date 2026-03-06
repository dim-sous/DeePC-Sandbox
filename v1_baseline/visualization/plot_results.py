"""Plotting utilities for DeePC trajectory tracking results."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.parameters import DeePCConfig


def plot_all(
    results: dict,
    config: DeePCConfig,
    save_dir: str = ".",
) -> None:
    """Generate a single combined figure with all result plots.

    Layout (3 rows x 2 cols):
        [0,0] 2-D trajectory (x-y plane)
        [0,1] Velocity tracking
        [1,0] x tracking error
        [1,1] y tracking error
        [2,0–1] Control inputs (steering + acceleration on one axis)
    """
    y_hist = results["y_history"]
    y_ref = results["y_ref_history"]
    u_hist = results["u_history"]
    times = results["times"]

    # Align lengths
    n = min(len(times), len(y_hist), len(y_ref), len(u_hist))
    times = times[:n]
    y_hist = y_hist[:n]
    y_ref = y_ref[:n]
    u_hist = u_hist[:n]

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # --- [0,0] 2-D trajectory ---
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_traj.plot(y_ref[:, 0], y_ref[:, 1], "r--", linewidth=1.5, label="Reference")
    ax_traj.plot(y_hist[:, 0], y_hist[:, 1], "b-", linewidth=1.2, label="Actual")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_title("Trajectory (x-y plane)")
    ax_traj.legend(fontsize=8)
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.grid(True, alpha=0.3)

    # --- [0,1] Velocity tracking ---
    ax_vel = fig.add_subplot(gs[0, 1])
    ax_vel.plot(times, y_ref[:, 2], "r--", linewidth=1.2, label="Reference")
    ax_vel.plot(times, y_hist[:, 2], "b-", linewidth=1.0, label="Actual")
    ax_vel.set_xlabel("Time [s]")
    ax_vel.set_ylabel("v [m/s]")
    ax_vel.set_title("Velocity Tracking")
    ax_vel.legend(fontsize=8)
    ax_vel.grid(True, alpha=0.3)

    # --- [1,0] x tracking ---
    ax_x = fig.add_subplot(gs[1, 0])
    ax_x.plot(times, y_ref[:, 0], "r--", linewidth=1.2, label="Reference")
    ax_x.plot(times, y_hist[:, 0], "b-", linewidth=1.0, label="Actual")
    ax_x.set_xlabel("Time [s]")
    ax_x.set_ylabel("x [m]")
    ax_x.set_title("Longitudinal Tracking")
    ax_x.legend(fontsize=8)
    ax_x.grid(True, alpha=0.3)

    # --- [1,1] y tracking ---
    ax_y = fig.add_subplot(gs[1, 1])
    ax_y.plot(times, y_ref[:, 1], "r--", linewidth=1.2, label="Reference")
    ax_y.plot(times, y_hist[:, 1], "b-", linewidth=1.0, label="Actual")
    ax_y.set_xlabel("Time [s]")
    ax_y.set_ylabel("y [m]")
    ax_y.set_title("Lateral Tracking")
    ax_y.legend(fontsize=8)
    ax_y.grid(True, alpha=0.3)

    # --- [2,0:1] Control inputs (combined) ---
    ax_u = fig.add_subplot(gs[2, :])
    color_steer = "tab:blue"
    color_accel = "tab:orange"

    ax_u.plot(times, u_hist[:, 0], color=color_steer, linewidth=0.9, label="Steering [rad]")
    ax_u.axhline(-config.delta_max, color=color_steer, linestyle=":", alpha=0.4)
    ax_u.axhline(config.delta_max, color=color_steer, linestyle=":", alpha=0.4)
    ax_u.set_ylabel("Steering [rad]", color=color_steer)
    ax_u.tick_params(axis="y", labelcolor=color_steer)

    ax_u2 = ax_u.twinx()
    ax_u2.plot(times, u_hist[:, 1], color=color_accel, linewidth=0.9, label="Acceleration [m/s^2]")
    ax_u2.axhline(config.a_min, color=color_accel, linestyle=":", alpha=0.4)
    ax_u2.axhline(config.a_max, color=color_accel, linestyle=":", alpha=0.4)
    ax_u2.set_ylabel("Acceleration [m/s^2]", color=color_accel)
    ax_u2.tick_params(axis="y", labelcolor=color_accel)

    ax_u.set_xlabel("Time [s]")
    ax_u.set_title("Control Inputs")
    ax_u.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax_u.get_legend_handles_labels()
    lines2, labels2 = ax_u2.get_legend_handles_labels()
    ax_u.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.suptitle("v1_baseline — DeePC Results", fontsize=14, fontweight="bold")
    fig.savefig(f"{save_dir}/v1_baseline_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {save_dir}/v1_baseline_results.png")
