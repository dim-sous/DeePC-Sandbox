"""v3 — DeePC with output scaling for QP conditioning.

Extends v2 by normalizing output channels before Hankel construction,
fixing the conditioning issues (K8, K10-K12).

Usage (from repo root):
    uv run python -m v3.main
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

VERSION_TAG = "v3"

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"

from config.parameters import DeePCConfig
from data.data_generation import collect_data
from deepc.deepc_controller import DeePCController
from plants.bicycle_model import BicycleModel
from visualization.plot_results import plot_all


def generate_reference_trajectory(config: DeePCConfig) -> np.ndarray:
    """Generate a demanding multi-phase reference trajectory.

    Segments (8 phases):
      1. Sine weave at cruise speed (warm-up)
      2. Hard lane change left + accelerate
      3. Slalom at high speed (increasing frequency)
      4. Emergency braking to low speed
      5. Tight S-curve at low speed
      6. Accelerate back + sweeping right turn
      7. Double lane change (left then right)
      8. Cruise with damped oscillation (settle)
    """
    total = config.Tini + config.sim_steps + config.N
    y_ref = np.zeros((total, config.p))

    # Phase boundaries (fractions of total)
    cuts = [0.0, 0.10, 0.20, 0.35, 0.42, 0.55, 0.65, 0.82, 1.0]
    idx = [int(c * total) for c in cuts]

    v_cruise = config.v_ref
    v_high = config.v_ref * 1.4
    v_low = config.v_ref * 0.6
    A = config.ref_amplitude
    f0 = config.ref_frequency

    for k in range(total):
        t = k * config.Ts

        if k < idx[1]:
            # Phase 1: sine weave at cruise
            y_ref[k, 1] = A * 0.5 * np.sin(2 * np.pi * f0 * t)
            y_ref[k, 2] = v_cruise

        elif k < idx[2]:
            # Phase 2: hard lane change left + accelerate
            blend = (k - idx[1]) / (idx[2] - idx[1])
            y_ref[k, 1] = A * 0.8 * (1 - np.cos(np.pi * blend)) / 2
            y_ref[k, 2] = v_cruise + (v_high - v_cruise) * blend

        elif k < idx[3]:
            # Phase 3: slalom at high speed (increasing frequency)
            blend = (k - idx[2]) / (idx[3] - idx[2])
            freq = f0 * (1 + 2 * blend)  # frequency ramps up 1x to 3x
            offset = A * 0.8
            y_ref[k, 1] = offset + A * 0.6 * np.sin(2 * np.pi * freq * t)
            y_ref[k, 2] = v_high

        elif k < idx[4]:
            # Phase 4: emergency braking
            blend = (k - idx[3]) / (idx[4] - idx[3])
            y_ref[k, 1] = A * 0.8  # hold lateral
            y_ref[k, 2] = v_high + (v_low - v_high) * blend

        elif k < idx[5]:
            # Phase 5: tight S-curve at low speed
            blend = (k - idx[4]) / (idx[5] - idx[4])
            y_ref[k, 1] = A * 0.8 * np.cos(2 * np.pi * blend)
            y_ref[k, 2] = v_low

        elif k < idx[6]:
            # Phase 6: accelerate + sweeping right turn
            blend = (k - idx[5]) / (idx[6] - idx[5])
            y_ref[k, 1] = -A * 0.8 + A * 0.4 * blend
            y_ref[k, 2] = v_low + (v_cruise - v_low) * blend

        elif k < idx[7]:
            # Phase 7: double lane change (left-right-center)
            blend = (k - idx[6]) / (idx[7] - idx[6])
            y_ref[k, 1] = -A * 0.4 + A * np.sin(2 * np.pi * blend)
            y_ref[k, 2] = v_cruise

        else:
            # Phase 8: settle with damped oscillation
            blend = (k - idx[7]) / max(total - idx[7], 1)
            decay = np.exp(-3 * blend)
            y_ref[k, 1] = A * 0.3 * decay * np.sin(2 * np.pi * f0 * 3 * t)
            y_ref[k, 2] = v_cruise

        # x integrates velocity
        if k > 0:
            y_ref[k, 0] = y_ref[k - 1, 0] + y_ref[k, 2] * config.Ts

    return y_ref


def run_deepc_simulation(config: DeePCConfig) -> dict:
    """Execute the full DeePC closed-loop experiment."""
    print("Collecting persistently exciting data...")
    u_data, y_data = collect_data(config, seed=42)
    print(f"  Data collected: u {u_data.shape}, y {y_data.shape}")

    print("Building DeePC controller...")
    controller = DeePCController(config, u_data, y_data)
    print("  Controller ready.")

    y_ref_full = generate_reference_trajectory(config)

    x0 = y_ref_full[0, 0]
    y0 = y_ref_full[0, 1]
    sim = BicycleModel(
        Ts=config.Ts,
        L_wheelbase=config.L_wheelbase,
        delta_max=config.delta_max,
        a_max=config.a_max,
        a_min=config.a_min,
        initial_state=np.array([x0, y0, 0.0, config.v_ref]),
    )

    u_buffer: list[np.ndarray] = []
    y_buffer: list[np.ndarray] = []
    for _ in range(config.Tini):
        u_k = np.zeros(config.m)
        y_k = sim.step(u_k)
        u_buffer.append(u_k)
        y_buffer.append(y_k)

    y_history = list(y_buffer)
    u_history = list(u_buffer)
    costs: list[float] = []
    sigma_norms: list[float] = []
    sigma_out_norms: list[float] = []
    statuses: list[str] = []
    solve_times: list[float] = []

    print(f"Running closed-loop simulation ({config.sim_steps} steps)...")
    for k in range(config.sim_steps):
        step_idx = k + config.Tini
        u_ini = np.array(u_buffer[-config.Tini:])
        y_ini = np.array(y_buffer[-config.Tini:])
        y_ref_horizon = y_ref_full[step_idx : step_idx + config.N]

        u_prev = u_buffer[-1]

        t_start = time.perf_counter()
        u_opt, info = controller.solve(u_ini, y_ini, y_ref_horizon, u_prev=u_prev)
        t_solve = time.perf_counter() - t_start

        y_new = sim.step(u_opt)

        u_buffer.append(u_opt)
        y_buffer.append(y_new)

        u_history.append(u_opt)
        y_history.append(y_new)
        costs.append(info["cost"])
        sigma_norms.append(info["sigma_y_norm"])
        sigma_out_norms.append(info["sigma_out_norm"])
        statuses.append(info["status"])
        solve_times.append(t_solve)

        if (k + 1) % 50 == 0 or k == 0:
            print(
                f"  Step {k + 1:>4d}/{config.sim_steps}  "
                f"status={info['status']}  "
                f"cost={info['cost']:.2f}  "
                f"solve={t_solve * 1000:.1f}ms"
            )

    total = config.Tini + config.sim_steps
    results = {
        "times": np.arange(total) * config.Ts,
        "y_history": np.array(y_history),
        "u_history": np.array(u_history),
        "y_ref_history": y_ref_full[:total],
        "costs": costs,
        "sigma_norms": sigma_norms,
        "sigma_out_norms": sigma_out_norms,
        "statuses": statuses,
        "solve_times": solve_times,
    }

    optimal_count = sum(1 for s in statuses if "optimal" in s)
    print(
        f"\nDone. Optimal solves: {optimal_count}/{len(statuses)}  "
        f"({100 * optimal_count / len(statuses):.0f}%)"
    )
    return results


def save_results(results: dict, version_tag: str) -> None:
    """Save simulation results to results/ directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    array_keys = ["times", "y_history", "u_history", "y_ref_history"]
    np.savez(
        RESULTS_DIR / f"{version_tag}_results.npz",
        **{k: results[k] for k in array_keys},
    )

    scalars = {
        "costs": results["costs"],
        "sigma_norms": [
            float(s) if s is not None else None for s in results["sigma_norms"]
        ],
        "sigma_out_norms": [
            float(s) if s is not None else None for s in results["sigma_out_norms"]
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

    save_results(results, VERSION_TAG)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_all(results, config, save_dir=str(RESULTS_DIR))

    sys.path.insert(0, str(REPO_ROOT))
    from comparison.metrics import compute_all_metrics, save_metrics

    metrics = compute_all_metrics(results, VERSION_TAG)
    save_metrics(metrics, RESULTS_DIR / f"{VERSION_TAG}_metrics.json")

    print("Experiment complete.")


if __name__ == "__main__":
    main()
