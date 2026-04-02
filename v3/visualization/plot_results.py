"""Plotting utilities for v3 DeePC results."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config.parameters import DeePCConfig

_STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.3,
    "legend.fontsize": 8,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.7",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}

_C_REF = "#d62728"
_C_ACT = "#1f77b4"
_C_ERR = "#2ca02c"
_C_STEER = "#1f77b4"
_C_ACCEL = "#ff7f0e"
_C_BAND = "#aaaaaa"


def plot_all(
    results: dict,
    config: DeePCConfig,
    save_dir: str = ".",
) -> None:
    """Generate v2 combined figure (3x2 layout).

    Row 0: trajectory (x-y), velocity tracking
    Row 1: tracking error, control inputs
    Row 2: input rates with constraint bands, output constraint bands
    """
    with plt.rc_context(_STYLE):
        _plot_impl(results, config, save_dir)


def _plot_impl(results: dict, config: DeePCConfig, save_dir: str) -> None:
    y_hist = np.asarray(results["y_history"])
    y_ref = np.asarray(results["y_ref_history"])
    u_hist = np.asarray(results["u_history"])
    times = np.asarray(results["times"])

    n = min(len(times), len(y_hist), len(y_ref), len(u_hist))
    times, y_hist, y_ref, u_hist = times[:n], y_hist[:n], y_ref[:n], u_hist[:n]

    pos_err = np.sqrt((y_hist[:, 0] - y_ref[:, 0]) ** 2 + (y_hist[:, 1] - y_ref[:, 1]) ** 2)
    vel_err = y_hist[:, 2] - y_ref[:, 2]
    rmse_pos = float(np.sqrt(np.mean(pos_err**2)))
    rmse_vel = float(np.sqrt(np.mean(vel_err**2)))
    max_pos_err = float(np.max(pos_err))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("v3 — DeePC Trajectory Tracking", fontsize=13, fontweight="bold", y=0.97)

    # ── [0,0] 2-D Trajectory ──────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(y_ref[:, 0], y_ref[:, 1], color=_C_REF, linestyle="--", linewidth=1.0, label="Reference")
    ax.plot(y_hist[:, 0], y_hist[:, 1], color=_C_ACT, linewidth=1.2, label="Actual")
    ax.plot(y_hist[0, 0], y_hist[0, 1], "o", color=_C_ACT, markersize=6, label="Start")
    ax.plot(y_hist[-1, 0], y_hist[-1, 1], "s", color=_C_ACT, markersize=6, label="End")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory (x-y)")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="datalim")
    ax.text(
        0.98, 0.02,
        f"RMSE$_{{pos}}$ = {rmse_pos:.3f} m\nmax err = {max_pos_err:.3f} m",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    # ── [0,1] Velocity Tracking ───────────────────────────────────
    ax = axes[0, 1]
    ax.plot(times, y_ref[:, 2], color=_C_REF, linestyle="--", linewidth=1.0, label="Reference")
    ax.plot(times, y_hist[:, 2], color=_C_ACT, linewidth=1.2, label="Actual")
    ax.fill_between(times, y_ref[:, 2] - np.abs(vel_err), y_ref[:, 2] + np.abs(vel_err),
                     color=_C_ERR, alpha=0.15, label="Error band")
    # Output constraint bands for velocity (channel 2)
    if np.isfinite(config.y_lb[2]):
        ax.axhline(config.y_lb[2], color=_C_BAND, linestyle="-.", linewidth=0.8, label=f"v_lb={config.y_lb[2]}")
    if np.isfinite(config.y_ub[2]):
        ax.axhline(config.y_ub[2], color=_C_BAND, linestyle="-.", linewidth=0.8, label=f"v_ub={config.y_ub[2]}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("v [m/s]")
    ax.set_title("Velocity Tracking + Output Bounds")
    ax.legend(loc="lower right")

    # ── [1,0] Position Tracking Error ─────────────────────────────
    ax = axes[1, 0]
    ax.fill_between(times, 0, pos_err, color=_C_ERR, alpha=0.3)
    ax.plot(times, pos_err, color=_C_ERR, linewidth=1.0)
    ax.axhline(rmse_pos, color=_C_ERR, linestyle=":", linewidth=0.8, alpha=0.7, label=f"RMSE = {rmse_pos:.3f} m")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position error [m]")
    ax.set_title("Tracking Error")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    # ── [1,1] Control Inputs ──────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(times, u_hist[:, 0], color=_C_STEER, linewidth=1.0, label="Steering $\\delta$ [rad]")
    ax.axhline(-config.delta_max, color=_C_STEER, linestyle=":", linewidth=0.7, alpha=0.5)
    ax.axhline(config.delta_max, color=_C_STEER, linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("Steering $\\delta$ [rad]", color=_C_STEER)
    ax.tick_params(axis="y", labelcolor=_C_STEER)

    ax2 = ax.twinx()
    ax2.plot(times, u_hist[:, 1], color=_C_ACCEL, linewidth=1.0, label="Accel $a$ [m/s$^2$]")
    ax2.axhline(config.a_min, color=_C_ACCEL, linestyle=":", linewidth=0.7, alpha=0.5)
    ax2.axhline(config.a_max, color=_C_ACCEL, linestyle=":", linewidth=0.7, alpha=0.5)
    ax2.set_ylabel("Accel $a$ [m/s$^2$]", color=_C_ACCEL)
    ax2.tick_params(axis="y", labelcolor=_C_ACCEL)
    ax.set_xlabel("Time [s]")
    ax.set_title("Control Inputs")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = f"{save_dir}/v3_results.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Plot saved to {path}")
