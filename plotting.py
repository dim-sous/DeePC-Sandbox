"""Plotting utilities for DeePC trajectory tracking results."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DeePCConfig


def plot_trajectory_2d(
    y_history: np.ndarray,
    y_ref_history: np.ndarray,
) -> plt.Figure:
    """Plot the 2-D (x–y) trajectory: actual vs reference.

    Args:
        y_history: Actual outputs, shape (T, p).
        y_ref_history: Reference outputs, shape (T, p).

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_ref_history[:, 0], y_ref_history[:, 1],
            "r--", linewidth=1.5, label="Reference")
    ax.plot(y_history[:, 0], y_history[:, 1],
            "b-", linewidth=1.2, label="Actual")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Vehicle Trajectory (x–y plane)")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_tracking_errors(
    times: np.ndarray,
    y_history: np.ndarray,
    y_ref_history: np.ndarray,
) -> plt.Figure:
    """Plot per-output tracking error over time.

    Args:
        times: Time vector, shape (T,).
        y_history: Actual outputs, shape (T, p).
        y_ref_history: Reference outputs, shape (T, p).

    Returns:
        Matplotlib Figure.
    """
    labels = ["x [m]", "y [m]", "v [m/s]"]
    p = y_history.shape[1]

    fig, axes = plt.subplots(p, 1, figsize=(10, 2.5 * p), sharex=True)
    if p == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(times, y_ref_history[:, i], "r--", label="Reference")
        ax.plot(times, y_history[:, i], "b-", label="Actual")
        ax.set_ylabel(labels[i] if i < len(labels) else f"y[{i}]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Output Tracking")
    fig.tight_layout()
    return fig


def plot_inputs(
    times: np.ndarray,
    u_history: np.ndarray,
    config: DeePCConfig,
) -> plt.Figure:
    """Plot control inputs over time with constraint bounds.

    Args:
        times: Time vector, shape (T,).
        u_history: Applied inputs, shape (T, m).
        config: Configuration (for constraint bounds).

    Returns:
        Matplotlib Figure.
    """
    labels = ["Steering δ [rad]", "Acceleration a [m/s²]"]
    bounds = [
        (-config.delta_max, config.delta_max),
        (config.a_min, config.a_max),
    ]
    m = u_history.shape[1]

    fig, axes = plt.subplots(m, 1, figsize=(10, 2.5 * m), sharex=True)
    if m == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(times, u_history[:, i], "g-", linewidth=1.0)
        lo, hi = bounds[i] if i < len(bounds) else (-1, 1)
        ax.axhline(lo, color="k", linestyle=":", alpha=0.5)
        ax.axhline(hi, color="k", linestyle=":", alpha=0.5)
        ax.set_ylabel(labels[i] if i < len(labels) else f"u[{i}]")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Control Inputs")
    fig.tight_layout()
    return fig


def plot_all(
    results: dict,
    config: DeePCConfig,
    save_dir: str = ".",
) -> None:
    """Generate and save all result plots.

    Args:
        results: Dictionary returned by ``run_deepc_simulation``.
        config: Configuration object.
        save_dir: Directory to save PNG files.
    """
    y_hist = results["y_history"]
    y_ref = results["y_ref_history"]
    u_hist = results["u_history"]
    times = results["times"]

    # Align lengths (trim to shortest)
    n = min(len(times), len(y_hist), len(y_ref), len(u_hist))
    times = times[:n]
    y_hist = y_hist[:n]
    y_ref = y_ref[:n]
    u_hist = u_hist[:n]

    fig1 = plot_trajectory_2d(y_hist, y_ref)
    fig1.savefig(f"{save_dir}/trajectory_2d.png", dpi=150)

    fig2 = plot_tracking_errors(times, y_hist, y_ref)
    fig2.savefig(f"{save_dir}/tracking_errors.png", dpi=150)

    fig3 = plot_inputs(times, u_hist, config)
    fig3.savefig(f"{save_dir}/control_inputs.png", dpi=150)

    plt.close("all")
    print(f"Plots saved to {save_dir}/")
