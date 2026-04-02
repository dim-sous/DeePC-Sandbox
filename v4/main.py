"""v4 — DeePC with sparse QP form per original paper.

Sparse form with u, y as explicit decision variables. L1 norms
on g and sigma_y. No output scaling.

Usage (from repo root):
    uv run python -m v4.main
    uv run python -m v4.main --scenario hard|circle|square|zigzag
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

VERSION_TAG = "v4"

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results" / "v4"

from config.parameters import DeePCConfig
from data.data_generation import collect_data
from deepc.deepc_controller import DeePCController
from plants.bicycle_model import BicycleModel
from visualization.plot_results import plot_all


def generate_reference_trajectory(config: DeePCConfig) -> np.ndarray:
    """Generate a sinusoidal reference trajectory (same as v1/v2/v3)."""
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


def generate_hard_reference(config: DeePCConfig) -> np.ndarray:
    """Generate a driving-course trajectory with multiple maneuver phases.

    Phases:
        1. Straight + accelerate      5→8 m/s
        2. Slalom (tight zigzag)       8 m/s
        3. Decelerate into sharp turn  8→3 m/s, ~90° right
        4. Tight circle                4 m/s, radius ≈10 m
        5. Straighten + accelerate     4→7 m/s
        6. Double lane change          7 m/s (elk test)
        7. Cruise (settle)             7 m/s
    """
    dt = config.Ts
    total = config.Tini + config.sim_steps + config.N

    # Define speed and heading-rate profiles, then integrate (x, y)
    v_profile = np.zeros(total)
    hdot_profile = np.zeros(total)  # heading rate [rad/s]

    # Phase boundaries (in steps)
    p1 = 50   # straight + accelerate
    p2 = 130  # slalom
    p3 = 180  # decelerate + sharp turn
    p4 = 260  # tight circle
    p5 = 300  # straighten + accelerate
    p6 = 350  # double lane change
    # p7 = remainder: cruise

    for k in range(total):
        if k < p1:
            # Phase 1: straight, accelerate 5→8
            frac = k / p1
            v_profile[k] = 5.0 + 3.0 * frac
            hdot_profile[k] = 0.0

        elif k < p2:
            # Phase 2: slalom — aggressive sinusoidal heading changes
            v_profile[k] = 8.0
            t_phase = (k - p1) * dt
            hdot_profile[k] = 1.8 * np.sin(2 * np.pi * 0.5 * t_phase)

        elif k < p3:
            # Phase 3: decelerate 8→3, sharp right turn (~90°)
            frac = (k - p2) / (p3 - p2)
            v_profile[k] = 8.0 - 5.0 * frac
            # Concentrate turn in middle of phase
            turn_center = 0.5
            turn_width = 0.3
            gauss = np.exp(-((frac - turn_center) ** 2) / (2 * turn_width ** 2))
            hdot_profile[k] = -3.0 * gauss  # right turn

        elif k < p4:
            # Phase 4: tight circle, radius ≈ 10m
            v_profile[k] = 4.0
            hdot_profile[k] = v_profile[k] / 10.0  # omega = v/R

        elif k < p5:
            # Phase 5: straighten out + accelerate 4→7
            frac = (k - p4) / (p5 - p4)
            v_profile[k] = 4.0 + 3.0 * frac
            # Ease out of circle turn
            hdot_profile[k] = 0.4 * (1.0 - frac)

        elif k < p6:
            # Phase 6: double lane change (elk test)
            v_profile[k] = 7.0
            t_phase = (k - p5) * dt
            period = (p6 - p5) * dt
            hdot_profile[k] = 1.5 * np.sin(2 * np.pi * (2.0 / period) * t_phase)

        else:
            # Phase 7: cruise
            v_profile[k] = 7.0
            hdot_profile[k] = 0.0

    # Integrate heading and position
    heading = np.zeros(total)
    x = np.zeros(total)
    y = np.zeros(total)

    for k in range(1, total):
        heading[k] = heading[k - 1] + hdot_profile[k - 1] * dt
        x[k] = x[k - 1] + v_profile[k - 1] * np.cos(heading[k - 1]) * dt
        y[k] = y[k - 1] + v_profile[k - 1] * np.sin(heading[k - 1]) * dt

    y_ref = np.zeros((total, config.p))
    y_ref[:, 0] = x
    y_ref[:, 1] = y
    y_ref[:, 2] = v_profile

    return y_ref


def generate_circle_reference(config: DeePCConfig, radius: float = 15.0) -> np.ndarray:
    """Generate a circular trajectory at constant speed."""
    dt = config.Ts
    total = config.Tini + config.sim_steps + config.N
    v = config.v_ref
    omega = v / radius  # heading rate for circular motion

    heading = np.zeros(total)
    x = np.zeros(total)
    y = np.zeros(total)

    for k in range(1, total):
        heading[k] = heading[k - 1] + omega * dt
        x[k] = x[k - 1] + v * np.cos(heading[k - 1]) * dt
        y[k] = y[k - 1] + v * np.sin(heading[k - 1]) * dt

    y_ref = np.zeros((total, config.p))
    y_ref[:, 0] = x
    y_ref[:, 1] = y
    y_ref[:, 2] = v
    return y_ref


def generate_square_reference(config: DeePCConfig, side: float = 20.0) -> np.ndarray:
    """Generate a square trajectory: straight segments with 90-degree turns."""
    dt = config.Ts
    total = config.Tini + config.sim_steps + config.N
    v = config.v_ref

    # Time to traverse one side
    side_time = side / v
    side_steps = int(side_time / dt)
    # Time for a 90-degree turn (quarter circle, radius ~3m)
    turn_radius = 3.0
    turn_arc = (np.pi / 2) * turn_radius
    turn_steps = max(int(turn_arc / (v * dt)), 5)
    turn_omega = (np.pi / 2) / (turn_steps * dt)

    heading = np.zeros(total)
    x = np.zeros(total)
    y = np.zeros(total)
    v_profile = np.full(total, v)

    # Build repeating pattern: straight + turn
    leg_steps = side_steps + turn_steps
    for k in range(1, total):
        pos_in_leg = k % leg_steps
        if pos_in_leg < side_steps:
            # Straight segment
            hdot = 0.0
        else:
            # 90-degree left turn
            hdot = turn_omega

        heading[k] = heading[k - 1] + hdot * dt
        x[k] = x[k - 1] + v_profile[k - 1] * np.cos(heading[k - 1]) * dt
        y[k] = y[k - 1] + v_profile[k - 1] * np.sin(heading[k - 1]) * dt

    y_ref = np.zeros((total, config.p))
    y_ref[:, 0] = x
    y_ref[:, 1] = y
    y_ref[:, 2] = v_profile
    return y_ref


def generate_zigzag_reference(config: DeePCConfig, amplitude: float = 5.0, seg_length: float = 15.0) -> np.ndarray:
    """Generate a zigzag trajectory: alternating diagonal segments."""
    dt = config.Ts
    total = config.Tini + config.sim_steps + config.N
    v = config.v_ref

    # Each segment: travel seg_length forward while moving ±amplitude laterally
    heading_angle = np.arctan2(amplitude, seg_length)
    seg_dist = np.sqrt(seg_length**2 + amplitude**2)
    seg_steps = int(seg_dist / (v * dt))
    seg_steps = max(seg_steps, 5)

    heading = np.zeros(total)
    x = np.zeros(total)
    y = np.zeros(total)

    # Transition steps for heading change between segments
    trans_steps = 5
    target_heading = heading_angle  # start going up-right

    for k in range(1, total):
        seg_idx = k // seg_steps
        pos_in_seg = k % seg_steps

        if seg_idx % 2 == 0:
            target_heading = heading_angle
        else:
            target_heading = -heading_angle

        if pos_in_seg < trans_steps:
            # Smooth transition to new heading
            frac = pos_in_seg / trans_steps
            blend = 0.5 * (1 - np.cos(np.pi * frac))
            prev_target = -heading_angle if seg_idx % 2 == 0 else heading_angle
            if seg_idx == 0:
                prev_target = 0.0
            heading[k] = prev_target + blend * (target_heading - prev_target)
        else:
            heading[k] = target_heading

        x[k] = x[k - 1] + v * np.cos(heading[k]) * dt
        y[k] = y[k - 1] + v * np.sin(heading[k]) * dt

    y_ref = np.zeros((total, config.p))
    y_ref[:, 0] = x
    y_ref[:, 1] = y
    y_ref[:, 2] = v
    return y_ref


def run_deepc_simulation(
    config: DeePCConfig,
    y_ref_override: np.ndarray | None = None,
) -> dict:
    """Execute the full DeePC closed-loop experiment."""
    print("Collecting multi-regime training data...")
    u_data, y_data = collect_data(config, seed=42)
    print(f"  Data collected: u {u_data.shape}, y {y_data.shape}")

    print("Building DeePC controller...")
    controller = DeePCController(config, u_data, y_data)
    print("  Controller ready.")

    y_ref_full = y_ref_override if y_ref_override is not None else generate_reference_trajectory(config)

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
    parser = argparse.ArgumentParser(description="v4 DeePC simulation")
    parser.add_argument(
        "--scenario", type=str, default="default",
        choices=["default", "hard", "circle", "square", "zigzag"],
        help="Reference trajectory scenario",
    )
    args = parser.parse_args()

    scenario = args.scenario

    if scenario == "hard":
        config = DeePCConfig(sim_steps=400)
        tag = f"{VERSION_TAG}_hard"
        print("=== HARD SCENARIO: driving course ===")
        y_ref = generate_hard_reference(config)
        results = run_deepc_simulation(config, y_ref_override=y_ref)
    elif scenario == "circle":
        config = DeePCConfig(sim_steps=300)
        tag = f"{VERSION_TAG}_circle"
        print("=== CIRCLE SCENARIO ===")
        y_ref = generate_circle_reference(config)
        results = run_deepc_simulation(config, y_ref_override=y_ref)
    elif scenario == "square":
        config = DeePCConfig(sim_steps=350)
        tag = f"{VERSION_TAG}_square"
        print("=== SQUARE SCENARIO ===")
        y_ref = generate_square_reference(config)
        results = run_deepc_simulation(config, y_ref_override=y_ref)
    elif scenario == "zigzag":
        config = DeePCConfig(sim_steps=300)
        tag = f"{VERSION_TAG}_zigzag"
        print("=== ZIGZAG SCENARIO ===")
        y_ref = generate_zigzag_reference(config)
        results = run_deepc_simulation(config, y_ref_override=y_ref)
    else:
        config = DeePCConfig()
        tag = VERSION_TAG
        results = run_deepc_simulation(config)

    save_results(results, tag)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_all(results, config, save_dir=str(RESULTS_DIR), tag=tag)

    sys.path.insert(0, str(REPO_ROOT))
    from comparison.metrics import compute_all_metrics, save_metrics

    metrics = compute_all_metrics(results, tag)
    save_metrics(metrics, RESULTS_DIR / f"{tag}_metrics.json")

    print("Experiment complete.")


if __name__ == "__main__":
    main()
