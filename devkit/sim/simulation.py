"""DeePC closed-loop simulation runner."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from deepc.config import DeePCConfig
from deepc.controller import DeePCController, constraints_to_arrays
from deepc.noise_estimator import NoiseEstimator
from plants.base import PlantBase


def solve_counts(results: dict) -> dict[str, int]:
    """Every solve of a run counted by its outcome, the one formula the loop's printed line, the
    closed-loop piece and ``sim.metrics.compute_metrics`` read: ``solves_<status>`` per status, matched
    exactly (D2); ``solves_by_<solver>`` per solver that answered; ``zero_inputs_applied``, the
    solves that left no solution, after which the controller's zero input was applied (D3)."""
    statuses, solvers = list(results["statuses"]), list(results["solvers"])
    out = {f"solves_{s}": statuses.count(s) for s in dict.fromkeys(statuses)}
    out.update({f"solves_by_{s}": solvers.count(s) for s in dict.fromkeys(solvers)})
    out["zero_inputs_applied"] = int(sum(results["zero_input"]))
    return out


@dataclass
class FeatureFlags:
    """Optional DeePC features enabled via CLI flags."""

    noise_adaptive: bool = False
    online_hankel: bool = False
    no_constraints: bool = False


def run_simulation(
    plant: PlantBase,
    config: DeePCConfig,
    features: FeatureFlags,
    u_data: np.ndarray,
    y_data: np.ndarray,
    path: np.ndarray,
) -> dict:
    """Execute the full DeePC closed-loop experiment.

    Parameters
    ----------
    plant : Instantiated plant model (will be reset internally).
    config : Algorithm configuration.
    features : Feature flags.
    u_data, y_data : Pre-collected training data.
    path : Reference trajectory from the plant's scenario generator.
    """
    print("Building DeePC controller...")
    constraints = plant.get_constraints()
    u_lb, u_ub, du_min, du_max, y_lb, y_ub = constraints_to_arrays(
        constraints, plant.input_names, plant.output_names,
    )
    if features.no_constraints:
        u_lb = np.full(plant.m, -np.inf)
        u_ub = np.full(plant.m, np.inf)
        du_min = np.full(plant.m, -np.inf)
        du_max = np.full(plant.m, np.inf)
        y_lb = np.full(plant.p, -np.inf)
        y_ub = np.full(plant.p, np.inf)
    controller = DeePCController(
        config, u_data, y_data, u_lb, u_ub, du_min, du_max, y_lb, y_ub,
    )
    print(f"  Controller ready. Rank of H_L(u): {controller.excitation_rank} of "
          f"{controller.excitation_rank_required} (L * m).")

    # Initialize plant at start of reference path
    initial_state = plant.get_initial_state_for_scenario(path[0])
    plant.reset(initial_state)

    # DeePC reference is always zero error
    zero_ref = np.zeros(config.p)

    # Tini warm-up: drive with zero input, record errors
    u_buffer: list[np.ndarray] = []
    y_buffer: list[np.ndarray] = []

    for k in range(config.Tini):
        u_k = np.zeros(config.m)
        plant.step(u_k)
        errors = plant.get_output(plant.state, path[k])
        u_buffer.append(u_k)
        y_buffer.append(errors)

    y_history = list(y_buffer)
    u_history = list(u_buffer)
    costs: list[float] = []
    sigma_norms: list[float] = []
    sigma_out_norms: list[float] = []
    statuses: list[str] = []
    solvers: list[str] = []
    zero_input: list[bool] = []
    solve_times: list[float] = []
    lambda_g_history: list[float] = []

    noise_est = NoiseEstimator(p=config.p, window=config.noise_estimation_window)
    y_predicted_prev: np.ndarray | None = None

    # Build progress label from first two output names
    progress_names = plant.output_names[:2]

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

        y_ref_horizon = np.tile(zero_ref, (config.N, 1))

        u_prev = u_buffer[-1]

        t_start = time.perf_counter()
        u_opt, info = controller.solve(u_ini, y_ini, y_ref_horizon, u_prev=u_prev)
        t_solve = time.perf_counter() - t_start

        if info["y_predicted"] is not None:
            y_predicted_prev = info["y_predicted"][0]
        else:
            y_predicted_prev = None

        plant.step(u_opt)

        errors = plant.get_output(plant.state, path[step_idx])

        # Online Hankel update
        if features.online_hankel and k >= config.hankel_warmup_steps:
            noise_std = noise_est.get_noise_std()
            mean_residual = float(np.mean(noise_std))
            if mean_residual < 1.0:
                controller.update_hankel(u_opt, errors)

        u_buffer.append(u_opt)
        y_buffer.append(errors)

        u_history.append(u_opt)
        y_history.append(errors)
        costs.append(info["cost"])
        sigma_norms.append(info["sigma_y_norm"])
        sigma_out_norms.append(info["sigma_out_norm"])
        statuses.append(info["status"])
        solvers.append(info["solver"])
        zero_input.append(bool(info["no_solution"]))
        solve_times.append(t_solve)
        lambda_g_history.append(info["lambda_g"])

        if (k + 1) % 50 == 0 or k == 0:
            err_strs = "  ".join(
                f"{n}={errors[i]:.3f}" for i, n in enumerate(progress_names)
            )
            print(
                f"  Step {k + 1:>4d}/{config.sim_steps}  "
                f"status={info['status']}  "
                f"cost={'none' if info['cost'] is None else format(info['cost'], '.2f')}  "
                f"solve={t_solve * 1000:.1f}ms  "
                f"{err_strs}"
            )

    total = config.Tini + config.sim_steps

    results: dict = {
        "times": np.arange(total) * config.Ts,
        "y_history": np.array(y_history),
        "u_history": np.array(u_history),
        "ref_path": path[:total],
        "costs": costs,
        "sigma_norms": sigma_norms,
        "sigma_out_norms": sigma_out_norms,
        "statuses": statuses,
        "solvers": solvers,
        "zero_input": zero_input,
        "excitation_rank": controller.excitation_rank,
        "excitation_rank_required": controller.excitation_rank_required,
        "solve_times": solve_times,
        "lambda_g_history": lambda_g_history,
    }

    print(f"\nDone. {len(statuses)} solves: "
          + ", ".join(f"{k} {v}" for k, v in solve_counts(results).items()))
    return results
