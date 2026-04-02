"""DeePC closed-loop simulation runner."""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass

import numpy as np

from control.config import DeePCConfig
from control.controller import DeePCController
from control.noise_estimator import NoiseEstimator
from sim.eval.data_generation import collect_data
from sim.eval.scenarios import generate_reference_trajectory
from plants.bicycle_model import BicycleModel


@dataclass
class FeatureFlags:
    """Optional DeePC features enabled via CLI flags."""

    noise_adaptive: bool = False
    online_hankel: bool = False


def run_simulation(
    config: DeePCConfig,
    features: FeatureFlags,
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
    lambda_g_history: list[float] = []

    noise_est = NoiseEstimator(p=config.p, window=config.noise_estimation_window)
    y_predicted_prev: np.ndarray | None = None

    print(f"Running closed-loop simulation ({config.sim_steps} steps)...")
    for k in range(config.sim_steps):
        step_idx = k + config.Tini

        # Noise-adaptive regularization (v5 feature)
        if features.noise_adaptive and y_predicted_prev is not None:
            y_actual = y_buffer[-1]
            noise_est.update(y_predicted_prev, y_actual)

            scaling = noise_est.get_scaling_factor(
                config.baseline_noise_std, config.max_lambda_scaling
            )
            lambda_g_adaptive = config.lambda_g * scaling
            lambda_y_adaptive = config.lambda_y * scaling

            noise_std = noise_est.get_noise_std()
            tightening = np.tile(
                noise_std * config.constraint_tightening_factor, config.N
            )
            controller.update_robustness(lambda_g_adaptive, lambda_y_adaptive, tightening)

        u_ini = np.array(u_buffer[-config.Tini:])
        y_ini = np.array(y_buffer[-config.Tini:])
        y_ref_horizon = y_ref_full[step_idx : step_idx + config.N]

        u_prev = u_buffer[-1]

        t_start = time.perf_counter()
        u_opt, info = controller.solve(u_ini, y_ini, y_ref_horizon, u_prev=u_prev)
        t_solve = time.perf_counter() - t_start

        # Store one-step-ahead prediction for next noise update
        if info["y_predicted"] is not None:
            y_predicted_prev = info["y_predicted"][0]
        else:
            y_predicted_prev = None

        y_new = sim.step(u_opt)

        # Online Hankel update (v6 feature)
        if features.online_hankel and k >= config.hankel_warmup_steps:
            noise_std = noise_est.get_noise_std()
            mean_residual = float(np.mean(noise_std))
            if mean_residual < 1.0:
                controller.update_hankel(u_opt, y_new)

        u_buffer.append(u_opt)
        y_buffer.append(y_new)

        u_history.append(u_opt)
        y_history.append(y_new)
        costs.append(info["cost"])
        sigma_norms.append(info["sigma_y_norm"])
        sigma_out_norms.append(info["sigma_out_norm"])
        statuses.append(info["status"])
        solve_times.append(t_solve)
        lambda_g_history.append(info["lambda_g"])

        if (k + 1) % 50 == 0 or k == 0:
            print(
                f"  Step {k + 1:>4d}/{config.sim_steps}  "
                f"status={info['status']}  "
                f"cost={info['cost']:.2f}  "
                f"solve={t_solve * 1000:.1f}ms  "
                f"lambda_g={info['lambda_g']:.1f}  "
                f"H_upd={info['hankel_updates']}"
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
        "lambda_g_history": lambda_g_history,
    }

    optimal_count = sum(1 for s in statuses if "optimal" in s)
    print(
        f"\nDone. Optimal solves: {optimal_count}/{len(statuses)}  "
        f"({100 * optimal_count / len(statuses):.0f}%)"
    )
    return results


def save_results(results: dict, tag: str, results_dir: pathlib.Path) -> None:
    """Save simulation results to results/ directory."""
    results_dir.mkdir(parents=True, exist_ok=True)

    array_keys = ["times", "y_history", "u_history", "y_ref_history"]
    np.savez(
        results_dir / f"{tag}_results.npz",
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
    with open(results_dir / f"{tag}_scalars.json", "w") as f:
        json.dump(scalars, f, indent=2)

    print(f"Results saved to {results_dir}/")
