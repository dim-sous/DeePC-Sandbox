"""v1_baseline — DeePC trajectory tracking for autonomous vehicle.

Baseline version with timing instrumentation and results saving.

Usage (from repo root):
    uv run python -m v1_baseline.main
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import time

import numpy as np

VERSION_TAG = "v1_baseline"

# Ensure version folder root is importable (and ONLY this folder)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPO_ROOT = PROJECT_ROOT.parent
RESULTS_DIR = REPO_ROOT / "results"

from config.parameters import DeePCConfig
from data.data_generation import collect_data
from deepc.deepc_controller import DeePCController
from simulation.vehicle_simulator import VehicleSimulator
from visualization.plot_results import plot_all


def generate_reference_trajectory(config: DeePCConfig) -> np.ndarray:
    """Generate a sinusoidal reference trajectory.

    The vehicle drives forward at constant velocity while following a
    sinusoidal lateral (y) path:

        x_ref(k) = v_ref * k * Ts
        y_ref(k) = A * sin(2pi f k Ts)
        v_ref(k) = v_ref

    The array is padded by N extra steps so the controller always has a
    full prediction horizon available.

    Args:
        config: Experiment configuration.

    Returns:
        Reference array, shape (Tini + sim_steps + N, p).
    """
    total = config.Tini + config.sim_steps + config.N
    y_ref = np.zeros((total, config.p))

    for k in range(total):
        t = k * config.Ts
        y_ref[k, 0] = config.v_ref * t
        y_ref[k, 1] = config.ref_amplitude * np.sin(
            2 * np.pi * config.ref_frequency * t
        )
        y_ref[k, 2] = config.v_ref

    return y_ref


def run_deepc_simulation(config: DeePCConfig) -> dict:
    """Execute the full DeePC closed-loop experiment.

    Steps:
        1. Collect persistently exciting offline data.
        2. Build the DeePC controller (Hankel matrices + CVXPY problem).
        3. Generate the reference trajectory.
        4. Initialise the simulator and warm-up buffers.
        5. Run the receding-horizon control loop.

    Args:
        config: Experiment configuration.

    Returns:
        Dictionary with keys ``times``, ``y_history``, ``u_history``,
        ``y_ref_history``, ``costs``, ``sigma_norms``, ``statuses``,
        ``solve_times``.
    """
    # --- Step 1: Collect offline data ---
    print("Collecting persistently exciting data...")
    u_data, y_data = collect_data(config, seed=42)
    print(f"  Data collected: u {u_data.shape}, y {y_data.shape}")

    # --- Step 2: Build controller ---
    print("Building DeePC controller...")
    controller = DeePCController(config, u_data, y_data)
    print("  Controller ready.")

    # --- Step 3: Reference trajectory ---
    y_ref_full = generate_reference_trajectory(config)

    # --- Step 4: Initialise simulator ---
    x0 = y_ref_full[0, 0]
    y0 = y_ref_full[0, 1]
    sim = VehicleSimulator(
        config,
        initial_state=np.array([x0, y0, 0.0, config.v_ref]),
    )

    # Warm-up: apply zero inputs for Tini steps to fill the past buffers
    u_buffer: list[np.ndarray] = []
    y_buffer: list[np.ndarray] = []

    for _ in range(config.Tini):
        u_k = np.zeros(config.m)
        y_k = sim.step(u_k)
        u_buffer.append(u_k)
        y_buffer.append(y_k)

    # --- Step 5: Receding-horizon loop ---
    y_history = list(y_buffer)
    u_history = list(u_buffer)
    costs: list[float] = []
    sigma_norms: list[float] = []
    statuses: list[str] = []
    solve_times: list[float] = []

    print(f"Running closed-loop simulation ({config.sim_steps} steps)...")
    for k in range(config.sim_steps):
        step_idx = k + config.Tini  # global time index

        # Past trajectory windows
        u_ini = np.array(u_buffer[-config.Tini:])  # (Tini, m)
        y_ini = np.array(y_buffer[-config.Tini:])  # (Tini, p)

        # Future reference window
        y_ref_horizon = y_ref_full[step_idx : step_idx + config.N]

        # Solve DeePC
        t_start = time.perf_counter()
        u_opt, info = controller.solve(u_ini, y_ini, y_ref_horizon)
        t_solve = time.perf_counter() - t_start

        # Apply to plant
        y_new = sim.step(u_opt)

        # Update buffers
        u_buffer.append(u_opt)
        y_buffer.append(y_new)

        # Record
        u_history.append(u_opt)
        y_history.append(y_new)
        costs.append(info["cost"])
        sigma_norms.append(info["sigma_y_norm"])
        statuses.append(info["status"])
        solve_times.append(t_solve)

        if (k + 1) % 50 == 0 or k == 0:
            print(
                f"  Step {k + 1:>4d}/{config.sim_steps}  "
                f"status={info['status']}  "
                f"cost={info['cost']:.2f}  "
                f"solve={t_solve*1000:.1f}ms"
            )

    # --- Package results ---
    total = config.Tini + config.sim_steps
    results = {
        "times": np.arange(total) * config.Ts,
        "y_history": np.array(y_history),
        "u_history": np.array(u_history),
        "y_ref_history": y_ref_full[:total],
        "costs": costs,
        "sigma_norms": sigma_norms,
        "statuses": statuses,
        "solve_times": solve_times,
    }

    # Summary
    optimal_count = sum(1 for s in statuses if "optimal" in s)
    print(
        f"\nDone. Optimal solves: {optimal_count}/{len(statuses)}  "
        f"({100 * optimal_count / len(statuses):.0f}%)"
    )
    return results


def save_results(results: dict, version_tag: str) -> None:
    """Save simulation results to results/ directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays
    array_keys = ["times", "y_history", "u_history", "y_ref_history"]
    np.savez(
        RESULTS_DIR / f"{version_tag}_results.npz",
        **{k: results[k] for k in array_keys},
    )

    # Save scalar/list data
    scalars = {
        "costs": results["costs"],
        "sigma_norms": [
            float(s) if s is not None else None
            for s in results["sigma_norms"]
        ],
        "statuses": results["statuses"],
        "solve_times": results["solve_times"],
    }
    with open(RESULTS_DIR / f"{version_tag}_scalars.json", "w") as f:
        json.dump(scalars, f, indent=2)

    print(f"Results saved to {RESULTS_DIR}/")


def main() -> None:
    """Entry point."""
    config = DeePCConfig()
    results = run_deepc_simulation(config)

    # Save results and plots
    save_results(results, VERSION_TAG)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_all(results, config, save_dir=str(RESULTS_DIR))

    # Compute and save metrics
    sys.path.insert(0, str(REPO_ROOT))
    from comparison.metrics import compute_all_metrics, save_metrics

    metrics = compute_all_metrics(results, VERSION_TAG)
    save_metrics(metrics, RESULTS_DIR / f"{VERSION_TAG}_metrics.json")

    print("Experiment complete.")


if __name__ == "__main__":
    main()
