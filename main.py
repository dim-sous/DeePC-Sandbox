"""Main entry point for the DeePC trajectory tracking experiment."""

from __future__ import annotations

import sys
import numpy as np

from config import DeePCConfig
from data.data_generation import collect_data
from deepc.deepc_controller import DeePCController
from simulator.vehicle_simulator import VehicleSimulator
from plotting import plot_all


def generate_reference_trajectory(config: DeePCConfig) -> np.ndarray:
    """Generate a sinusoidal reference trajectory.

    The vehicle drives forward at constant velocity while following a
    sinusoidal lateral (y) path:

        x_ref(k) = v_ref * k * Ts
        y_ref(k) = A * sin(2π f k Ts)
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
        ``y_ref_history``, ``costs``, ``sigma_norms``, ``statuses``.
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

    print(f"Running closed-loop simulation ({config.sim_steps} steps)...")
    for k in range(config.sim_steps):
        step_idx = k + config.Tini  # global time index

        # Past trajectory windows
        u_ini = np.array(u_buffer[-config.Tini:])  # (Tini, m)
        y_ini = np.array(y_buffer[-config.Tini:])  # (Tini, p)

        # Future reference window
        y_ref_horizon = y_ref_full[step_idx : step_idx + config.N]

        # Solve DeePC
        u_opt, info = controller.solve(u_ini, y_ini, y_ref_horizon)

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

        if (k + 1) % 50 == 0 or k == 0:
            print(
                f"  Step {k + 1:>4d}/{config.sim_steps}  "
                f"status={info['status']}  "
                f"cost={info['cost']:.2f}"
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
    }

    # Summary
    optimal_count = sum(1 for s in statuses if "optimal" in s)
    print(
        f"\nDone. Optimal solves: {optimal_count}/{len(statuses)}  "
        f"({100 * optimal_count / len(statuses):.0f}%)"
    )
    return results


def main() -> None:
    """Entry point."""
    config = DeePCConfig()
    results = run_deepc_simulation(config)
    plot_all(results, config, save_dir=".")
    print("Experiment complete.")


if __name__ == "__main__":
    main()
