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
from sim.data_generation import collect_data
from sim.scenarios import generate_reference_path
from plants.bicycle_model import BicycleModel, compute_path_errors


@dataclass
class FeatureFlags:
    """Optional DeePC features enabled via CLI flags."""

    noise_adaptive: bool = False
    online_hankel: bool = False


def run_simulation(
    config: DeePCConfig,
    features: FeatureFlags,
    path_override: np.ndarray | None = None,
) -> dict:
    """Execute the full DeePC closed-loop experiment.

    Parameters
    ----------
    path_override : (total, 4) array of [x, y, heading, v] waypoints.
        If None, generates the default reference path.
    """
    print("Collecting multi-regime training data...")
    u_data, y_data = collect_data(config, seed=42)
    print(f"  Data collected: u {u_data.shape}, y {y_data.shape}")

    print("Building DeePC controller...")
    controller = DeePCController(config, u_data, y_data)
    print("  Controller ready.")

    path = path_override if path_override is not None else generate_reference_path(config)

    # Initialize vehicle at start of reference path
    x0, y0, heading0, v0 = path[0]
    sim = BicycleModel(
        Ts=config.Ts,
        L_wheelbase=config.L_wheelbase,
        delta_max=config.delta_max,
        a_max=config.a_max,
        a_min=config.a_min,
        initial_state=np.array([x0, y0, heading0, v0]),
    )

    # DeePC reference is always zero error
    zero_ref = np.zeros(config.p)

    # Tini warm-up: drive with zero input, record errors
    u_buffer: list[np.ndarray] = []
    y_buffer: list[np.ndarray] = []
    pos_history: list[np.ndarray] = [sim.position]

    for k in range(config.Tini):
        u_k = np.zeros(config.m)
        sim.step(u_k)
        errors = compute_path_errors(sim.state, *path[k])
        u_buffer.append(u_k)
        y_buffer.append(errors)
        pos_history.append(sim.position)

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

        # Noise-adaptive regularization
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

        # Zero-error reference for the whole horizon
        y_ref_horizon = np.tile(zero_ref, (config.N, 1))

        u_prev = u_buffer[-1]

        t_start = time.perf_counter()
        u_opt, info = controller.solve(u_ini, y_ini, y_ref_horizon, u_prev=u_prev)
        t_solve = time.perf_counter() - t_start

        if info["y_predicted"] is not None:
            y_predicted_prev = info["y_predicted"][0]
        else:
            y_predicted_prev = None

        sim.step(u_opt)

        # Compute errors relative to reference path
        errors = compute_path_errors(sim.state, *path[step_idx])

        # Online Hankel update
        if features.online_hankel and k >= config.hankel_warmup_steps:
            noise_std = noise_est.get_noise_std()
            mean_residual = float(np.mean(noise_std))
            if mean_residual < 1.0:
                controller.update_hankel(u_opt, errors)

        u_buffer.append(u_opt)
        y_buffer.append(errors)
        pos_history.append(sim.position)

        u_history.append(u_opt)
        y_history.append(errors)
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
                f"e_lat={errors[0]:.3f}  "
                f"e_head={errors[1]:.3f}"
            )

    total = config.Tini + config.sim_steps

    # Reference absolute positions for plotting
    ref_pos = path[:total, :2]

    results = {
        "times": np.arange(total) * config.Ts,
        "y_history": np.array(y_history),       # errors [e_lat, e_head, e_v]
        "u_history": np.array(u_history),
        "pos_history": np.array(pos_history),   # absolute [x, y] per step
        "ref_pos_history": ref_pos,             # reference [x, y]
        "ref_path": path[:total],               # full [x, y, heading, v]
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
