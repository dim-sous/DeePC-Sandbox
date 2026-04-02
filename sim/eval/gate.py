"""Three-stage gate for the unified DeePC platform.

Tests structural correctness (A), tracking performance (B), and
robustness under stress (C) for each feature-flag combination.

Usage (from repo root):
    uv run python -m sim.gate
    uv run python -m sim.gate --combo base
    uv run python -m sim.gate --combo na+oh
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from control.config import DeePCConfig
from control.controller import DeePCController
from control.hankel import build_hankel_matrix, build_data_matrices
from control.noise_estimator import NoiseEstimator
from control.online_hankel import SlidingHankelWindow
from sim.eval.data_generation import collect_data
from sim.eval.simulation import FeatureFlags, run_simulation
from sim.eval.scenarios import generate_reference_trajectory
from sim.comparison.metrics import compute_all_metrics
from sim.comparison import stress_configs
from plants.bicycle_model import BicycleModel

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "gate"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

COMBOS = {
    "base": FeatureFlags(),
    "na": FeatureFlags(noise_adaptive=True),
    "oh": FeatureFlags(online_hankel=True),
    "na+oh": FeatureFlags(noise_adaptive=True, online_hankel=True),
}


def _tag(ok: bool) -> str:
    return PASS if ok else FAIL


def _generate_ref(config: DeePCConfig) -> np.ndarray:
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


def _compute_rmse_position(results: dict) -> float:
    y = results["y_history"]
    r = results["y_ref_history"]
    n = min(len(y), len(r))
    err = y[:n, :2] - r[:n, :2]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


# ====================================================================
#  STAGE A — Validation (structural correctness, run once)
# ====================================================================


def stage_a(config: DeePCConfig, u_data: np.ndarray, y_data: np.ndarray) -> list[dict]:
    checks: list[dict] = []

    L = config.L
    T = config.T_data
    m, p = config.m, config.p

    # A1: Hankel dimensions
    Hu = build_hankel_matrix(u_data, L)
    Hy = build_hankel_matrix(y_data, L)
    expected_cols = T - L + 1

    ok_hu = Hu.shape == (L * m, expected_cols)
    ok_hy = Hy.shape == (L * p, expected_cols)
    checks.append({"name": "A1  Hankel(u) dimensions", "passed": ok_hu,
                    "detail": f"expected ({L * m}, {expected_cols}), got {Hu.shape}"})
    checks.append({"name": "A1  Hankel(y) dimensions", "passed": ok_hy,
                    "detail": f"expected ({L * p}, {expected_cols}), got {Hy.shape}"})

    # A2: Partition consistency
    Up, Yp, Uf, Yf = build_data_matrices(u_data, y_data, config.Tini, config.N)
    ok_shapes = (Up.shape == (config.Tini * m, expected_cols)
                 and Yp.shape == (config.Tini * p, expected_cols)
                 and Uf.shape == (config.N * m, expected_cols)
                 and Yf.shape == (config.N * p, expected_cols))
    checks.append({"name": "A2  Partition shapes", "passed": ok_shapes,
                    "detail": f"Up {Up.shape}, Yp {Yp.shape}, Uf {Uf.shape}, Yf {Yf.shape}"})

    top_u = np.allclose(Up, Hu[:config.Tini * m, :])
    bot_u = np.allclose(Uf, Hu[config.Tini * m:, :])
    top_y = np.allclose(Yp, Hy[:config.Tini * p, :])
    bot_y = np.allclose(Yf, Hy[config.Tini * p:, :])
    checks.append({"name": "A2  Partitions match full Hankel", "passed": top_u and bot_u and top_y and bot_y,
                    "detail": f"Up={top_u}, Uf={bot_u}, Yp={top_y}, Yf={bot_y}"})

    # A3: Persistent excitation
    rank = np.linalg.matrix_rank(Hu, tol=1e-6)
    required = L * m
    checks.append({"name": "A3  Persistent excitation", "passed": rank >= required,
                    "detail": f"rank={rank}, required>={required}"})

    # A4: Simulator dynamics spot-check
    sim = BicycleModel(Ts=config.Ts, L_wheelbase=config.L_wheelbase,
                       delta_max=config.delta_max, a_max=config.a_max, a_min=config.a_min,
                       initial_state=np.array([0.0, 0.0, 0.0, 5.0]))
    x0, y0, th0, v0 = sim.state.copy()
    delta_test, a_test = 0.1, 0.5
    sim.step(np.array([delta_test, a_test]))
    x1, y1, th1, v1 = sim.state
    Ts, Lw = config.Ts, config.L_wheelbase
    sim_ok = (np.isclose(x1, x0 + v0 * np.cos(th0) * Ts)
              and np.isclose(y1, y0 + v0 * np.sin(th0) * Ts)
              and np.isclose(th1, th0 + (v0 / Lw) * np.tan(delta_test) * Ts)
              and np.isclose(v1, max(v0 + a_test * Ts, 0.0)))
    checks.append({"name": "A4  Simulator dynamics spot check", "passed": bool(sim_ok),
                    "detail": f"state={sim.state}"})

    # A5: Hidden heading
    sim2 = BicycleModel(Ts=config.Ts, L_wheelbase=config.L_wheelbase,
                        delta_max=config.delta_max, a_max=config.a_max, a_min=config.a_min,
                        initial_state=np.array([1.0, 2.0, 0.5, 3.0]))
    out = sim2.output
    checks.append({"name": "A5  Output hides heading (dim=3)", "passed": len(out) == 3 and np.isclose(out[2], 3.0),
                    "detail": f"output={out}"})

    # A6: QP solves to optimality
    controller = DeePCController(config, u_data, y_data)
    u_ini = np.zeros((config.Tini, m))
    y_ini = np.zeros((config.Tini, p))
    y_ref = np.zeros((config.N, p))
    u_opt, info = controller.solve(u_ini, y_ini, y_ref)

    checks.append({"name": "A6  QP solves to optimality", "passed": "optimal" in info["status"],
                    "detail": f"status={info['status']}"})
    checks.append({"name": "A6  Objective finite and non-negative",
                    "passed": info["cost"] is not None and np.isfinite(info["cost"]) and info["cost"] >= -1e-6,
                    "detail": f"cost={info['cost']}"})

    # A7: Input box constraints
    if info["u_predicted"] is not None:
        u_pred = info["u_predicted"]
        steer_ok = bool(np.all(np.abs(u_pred[:, 0]) <= config.delta_max + 1e-6))
        accel_ok = bool(np.all(u_pred[:, 1] <= config.a_max + 1e-6)
                        and np.all(u_pred[:, 1] >= config.a_min - 1e-6))
    else:
        steer_ok = accel_ok = False
    checks.append({"name": "A7  Predicted inputs satisfy box constraints",
                    "passed": steer_ok and accel_ok,
                    "detail": f"steering_ok={steer_ok}, accel_ok={accel_ok}"})

    # A8: Input rate constraints
    if info["u_predicted"] is not None:
        u_pred = info["u_predicted"]
        du = np.diff(u_pred, axis=0)
        rate_steer_ok = bool(np.all(np.abs(du[:, 0]) <= config.d_delta_max + 1e-6))
        rate_accel_ok = bool(np.all(np.abs(du[:, 1]) <= config.da_max + 1e-6))
    else:
        rate_steer_ok = rate_accel_ok = False
    checks.append({"name": "A8  Predicted inputs satisfy rate constraints",
                    "passed": rate_steer_ok and rate_accel_ok,
                    "detail": f"d_steer_ok={rate_steer_ok}, d_accel_ok={rate_accel_ok}"})

    # A9: Output constraint slack activates under tight bounds
    tight_cfg = DeePCConfig(y_lb=[-np.inf, -0.01, 4.99], y_ub=[np.inf, 0.01, 5.01])
    tight_ctrl = DeePCController(tight_cfg, u_data, y_data)
    _, tight_info = tight_ctrl.solve(u_ini, y_ini, y_ref)
    slack_activates = (tight_info["sigma_out_norm"] is not None
                       and tight_info["sigma_out_norm"] > 1e-8)
    checks.append({"name": "A9  Output slack activates under tight bounds",
                    "passed": bool(slack_activates),
                    "detail": f"sigma_out_norm={tight_info.get('sigma_out_norm')}"})

    # A10: L1 regularization smoke test
    l1_cfg = DeePCConfig(reg_norm_g="L1", reg_norm_sigma_y="L1")
    l1_ctrl = DeePCController(l1_cfg, u_data, y_data)
    _, l1_info = l1_ctrl.solve(u_ini, y_ini, y_ref)
    l1_ok = l1_info["status"] is not None and "optimal" in l1_info["status"]
    checks.append({"name": "A10 L1 regularization solves to optimality",
                    "passed": l1_ok, "detail": f"status={l1_info['status']}"})

    # A11: Noise estimator warm-up returns zeros
    ne = NoiseEstimator(p=config.p, window=config.noise_estimation_window)
    warmup_ok = (
        float(np.sum(ne.get_noise_std())) == 0.0
        and ne.get_scaling_factor(config.baseline_noise_std) == 1.0
    )
    ne.update(np.array([1.0, 2.0, 5.0]), np.array([1.1, 2.0, 5.0]))
    ne.update(np.array([1.0, 2.0, 5.0]), np.array([0.9, 2.0, 5.0]))
    still_warmup = ne.get_scaling_factor(config.baseline_noise_std) == 1.0
    warmup_ok = warmup_ok and still_warmup
    checks.append({"name": "A11 Noise estimator warm-up returns zeros",
                    "passed": warmup_ok,
                    "detail": f"noise_std={ne.get_noise_std()}, scaling={ne.get_scaling_factor(config.baseline_noise_std)}"})

    # A12: Adaptive lambda propagates to controller
    controller.update_robustness(
        lambda_g=config.lambda_g * 5.0,
        lambda_y=config.lambda_y * 5.0,
        tightening=np.zeros(config.N * config.p),
    )
    _, adapted_info = controller.solve(u_ini, y_ini, y_ref)
    adapt_ok = (
        adapted_info["lambda_g"] is not None
        and abs(adapted_info["lambda_g"] - config.lambda_g * 5.0) < 1e-6
        and "optimal" in adapted_info["status"]
    )
    controller.update_robustness(
        lambda_g=config.lambda_g,
        lambda_y=config.lambda_y,
        tightening=np.zeros(config.N * config.p),
    )
    checks.append({"name": "A12 Adaptive lambda propagates to controller",
                    "passed": adapt_ok,
                    "detail": f"lambda_g={adapted_info.get('lambda_g')}, status={adapted_info['status']}"})

    # A13: Online Hankel window dimensions
    hw = SlidingHankelWindow(u_data, y_data, config.Tini, config.N,
                             config.m, config.p, config.hankel_window_size)
    hw_n = hw.n_cols
    dim_ok = (hw.Up.shape == (config.Tini * m, hw_n)
              and hw.Uf.shape == (config.N * m, hw_n)
              and hw.Yp.shape == (config.Tini * p, hw_n)
              and hw.Yf.shape == (config.N * p, hw_n)
              and hw_n > 0)
    checks.append({"name": "A13 Online Hankel window dimensions",
                    "passed": dim_ok,
                    "detail": f"Up={hw.Up.shape}, Uf={hw.Uf.shape}, n_cols={hw.n_cols}"})

    # A14: Column slide works and QP still solves
    hw.update(np.zeros(config.m), np.zeros(config.p))
    hw.update(np.zeros(config.m), np.zeros(config.p))
    controller.update_hankel(np.zeros(config.m), np.zeros(config.p))
    _, slide_info = controller.solve(u_ini, y_ini, y_ref)
    slide_ok = slide_info["status"] is not None and "optimal" in slide_info["status"]
    checks.append({"name": "A14 QP solves after Hankel slide",
                    "passed": slide_ok,
                    "detail": f"status={slide_info['status']}, hankel_updates={slide_info.get('hankel_updates')}"})

    return checks


# ====================================================================
#  STAGE B — Evaluation (per feature combo)
# ====================================================================


def stage_b(config: DeePCConfig, features: FeatureFlags, combo_name: str) -> tuple[list[dict], dict]:
    results = run_simulation(config, features)
    metrics = compute_all_metrics(results, combo_name)

    checks: list[dict] = []

    checks.append({"name": f"B1  [{combo_name}] Position RMSE < 1.0 m",
                    "passed": metrics["rmse_position"] < 1.0,
                    "detail": f"rmse_position={metrics['rmse_position']:.4f} m"})
    checks.append({"name": f"B1  [{combo_name}] Lateral RMSE (y) < 0.5 m",
                    "passed": metrics["rmse_y"] < 0.5,
                    "detail": f"rmse_y={metrics['rmse_y']:.4f} m"})
    checks.append({"name": f"B1  [{combo_name}] Velocity RMSE < 0.5 m/s",
                    "passed": metrics["rmse_v"] < 0.5,
                    "detail": f"rmse_v={metrics['rmse_v']:.4f} m/s"})
    checks.append({"name": f"B1  [{combo_name}] Max position error < 1.5 m",
                    "passed": metrics["max_position_error"] < 1.5,
                    "detail": f"max_pos_err={metrics['max_position_error']:.4f} m"})

    checks.append({"name": f"B2  [{combo_name}] Total control effort < 50",
                    "passed": metrics["total_control_effort"] < 50.0,
                    "detail": f"total_effort={metrics['total_control_effort']:.2f}"})

    checks.append({"name": f"B3  [{combo_name}] Optimal solve rate >= 95%",
                    "passed": metrics["optimal_solve_pct"] >= 95.0,
                    "detail": f"optimal_pct={metrics['optimal_solve_pct']:.1f}%"})
    checks.append({"name": f"B3  [{combo_name}] Avg solve time < 0.1 s",
                    "passed": metrics["avg_solve_time_s"] < 0.1,
                    "detail": f"avg_solve={metrics['avg_solve_time_s']:.4f} s"})
    checks.append({"name": f"B3  [{combo_name}] Max solve time < 0.5 s",
                    "passed": metrics["max_solve_time_s"] < 0.5,
                    "detail": f"max_solve={metrics['max_solve_time_s']:.4f} s"})

    checks.append({"name": f"B4  [{combo_name}] Mean slack norm < 0.1",
                    "passed": metrics["mean_sigma_y_norm"] < 0.1,
                    "detail": f"mean_sigma={metrics['mean_sigma_y_norm']:.6f}"})

    # Stability
    y_hist = np.asarray(results["y_history"])
    y_ref = np.asarray(results["y_ref_history"])
    n = min(len(y_hist), len(y_ref))
    pos_err = np.sqrt((y_hist[:n, 0] - y_ref[:n, 0]) ** 2 + (y_hist[:n, 1] - y_ref[:n, 1]) ** 2)
    mid = n // 2
    rmse_first = float(np.sqrt(np.mean(pos_err[:mid] ** 2)))
    rmse_second = float(np.sqrt(np.mean(pos_err[mid:] ** 2)))
    stable = rmse_second < max(rmse_first * 2.0, 1.0)
    checks.append({"name": f"B5  [{combo_name}] No divergence (2nd half RMSE <= 2x 1st half)",
                    "passed": stable,
                    "detail": f"1st_half={rmse_first:.4f}, 2nd_half={rmse_second:.4f}"})

    # Output constraint slack
    sigma_out_norms = results.get("sigma_out_norms", [])
    if sigma_out_norms:
        mean_out = float(np.mean([s for s in sigma_out_norms if s is not None]))
    else:
        mean_out = 0.0
    checks.append({"name": f"B6  [{combo_name}] Mean output slack norm < 1.0",
                    "passed": mean_out < 1.0,
                    "detail": f"mean_sigma_out={mean_out:.6f}"})

    # Input rate constraint satisfaction
    u_hist = np.asarray(results["u_history"])
    du = np.diff(u_hist, axis=0)
    rate_steer = bool(np.all(np.abs(du[:, 0]) <= config.d_delta_max + 1e-4))
    rate_accel = bool(np.all(np.abs(du[:, 1]) <= config.da_max + 1e-4))
    checks.append({"name": f"B7  [{combo_name}] Input rate constraints satisfied",
                    "passed": rate_steer and rate_accel,
                    "detail": f"d_steer_ok={rate_steer}, d_accel_ok={rate_accel}"})

    return checks, metrics


# ====================================================================
#  STAGE C — Stress Testing (per feature combo)
# ====================================================================


def _run_stress(config_overrides: dict, features: FeatureFlags,
                y_ref_fn=None, disturbance_fn=None) -> dict:
    """Run a stress test with given config overrides and features."""
    config = DeePCConfig(**config_overrides)
    if y_ref_fn is not None:
        y_ref = y_ref_fn(config)
    else:
        y_ref = _generate_ref(config)

    u_data, y_data = collect_data(config, seed=42)
    controller = DeePCController(config, u_data, y_data)

    x0 = y_ref[0, 0]
    y0 = y_ref[0, 1]
    sim = BicycleModel(
        Ts=config.Ts, L_wheelbase=config.L_wheelbase,
        delta_max=config.delta_max, a_max=config.a_max, a_min=config.a_min,
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
    statuses: list[str] = []
    solve_times: list[float] = []

    noise_est = NoiseEstimator(p=config.p, window=config.noise_estimation_window)
    y_predicted_prev: np.ndarray | None = None

    for k in range(config.sim_steps):
        step_idx = k + config.Tini

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
        y_ref_horizon = y_ref[step_idx: step_idx + config.N]

        u_prev = u_buffer[-1]

        t0 = time.perf_counter()
        u_opt, info = controller.solve(u_ini, y_ini, y_ref_horizon, u_prev=u_prev)
        solve_times.append(time.perf_counter() - t0)

        if info["y_predicted"] is not None:
            y_predicted_prev = info["y_predicted"][0]
        else:
            y_predicted_prev = None

        if disturbance_fn is not None:
            u_opt = disturbance_fn(k, u_opt, sim)

        y_new = sim.step(u_opt)

        if features.online_hankel and k >= config.hankel_warmup_steps:
            ns = noise_est.get_noise_std()
            if float(np.mean(ns)) < 1.0:
                controller.update_hankel(u_opt, y_new)

        u_buffer.append(u_opt)
        y_buffer.append(y_new)
        u_history.append(u_opt)
        y_history.append(y_new)
        statuses.append(info["status"])

    total = config.Tini + config.sim_steps
    return {
        "y_history": np.array(y_history),
        "u_history": np.array(u_history),
        "y_ref_history": y_ref[:total],
        "statuses": statuses,
        "solve_times": solve_times,
    }


def stage_c(features: FeatureFlags, combo_name: str) -> list[dict]:
    def _test(name, config_overrides, threshold_rmse, threshold_opt=90.0,
              y_ref_fn=None, disturbance_fn=None, extra_check=None):
        t0 = time.perf_counter()
        try:
            results = _run_stress(config_overrides, features,
                                  y_ref_fn=y_ref_fn, disturbance_fn=disturbance_fn)
            rmse = _compute_rmse_position(results)
            optimal_pct = 100.0 * sum(1 for s in results["statuses"] if "optimal" in s) / len(results["statuses"])
            ok = rmse < threshold_rmse and optimal_pct > threshold_opt
            if extra_check is not None:
                ok = ok and extra_check(results)
            elapsed = time.perf_counter() - t0
            return {"name": f"{name} [{combo_name}]", "passed": ok,
                    "detail": f"RMSE={rmse:.1f}m  opt={optimal_pct:.0f}%  elapsed={elapsed:.1f}s"}
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return {"name": f"{name} [{combo_name}]", "passed": False,
                    "detail": f"CRASHED: {e} (elapsed={elapsed:.1f}s)"}

    def _step_ref_fn(config):
        total = config.Tini + config.sim_steps + config.N
        y_ref = np.zeros((total, config.p))
        for k in range(total):
            t = k * config.Ts
            y_ref[k, 0] = config.v_ref * t
            y_ref[k, 2] = config.v_ref
            if k >= total // 2:
                y_ref[k, 1] = 3.0
        return y_ref

    def _disturbance(k, u_opt, sim):
        if 40 <= k <= 60:
            sim.state[3] += 0.5
        return u_opt

    def _rate_check(results):
        config = DeePCConfig(**stress_configs.rate_constrained_aggressive())
        u_hist = results["u_history"]
        du = np.diff(u_hist, axis=0)
        return (bool(np.all(np.abs(du[:, 0]) <= config.d_delta_max + 1e-4))
                and bool(np.all(np.abs(du[:, 1]) <= config.da_max + 1e-4)))

    def _tight_check(results):
        config = DeePCConfig(**stress_configs.tight_constraints())
        u_hist = results["u_history"]
        return (bool(np.all(np.abs(u_hist[:, 0]) <= config.delta_max + 1e-6))
                and bool(np.all(u_hist[:, 1] <= config.a_max + 1e-6))
                and bool(np.all(u_hist[:, 1] >= config.a_min - 1e-6)))

    checks = [
        _test("C1  High measurement noise (10x)", stress_configs.high_noise(), 2.0),
        _test("C2  Aggressive reference", stress_configs.aggressive_reference(), 5.0),
        _test("C3  Reduced dataset (T=50)", stress_configs.reduced_data(), 2.0),
        _test("C4  Tight input constraints", stress_configs.tight_constraints(), 2.0,
              extra_check=_tight_check),
        _test("C5  Nonlinear regime (high speed)", stress_configs.nonlinear_regime(), 2.0),
        _test("C6  Step reference change", stress_configs.step_reference(), 10.0,
              y_ref_fn=_step_ref_fn),
        _test("C7  Disturbance rejection", stress_configs.disturbance_rejection(), 3.0,
              disturbance_fn=_disturbance),
        _test("C8  Long horizon (200 steps)", stress_configs.long_horizon(), 1.0),
        _test("C9  Rate-constrained aggressive ref", stress_configs.rate_constrained_aggressive(), 5.0,
              extra_check=_rate_check),
    ]

    return checks


# ====================================================================
#  Report
# ====================================================================


def print_report(
    a_checks: list[dict],
    b_checks: list[dict],
    c_checks: list[dict],
    metrics: dict,
    wall_time: float,
    combos_run: list[str],
) -> dict:
    all_checks = a_checks + b_checks + c_checks
    total = len(all_checks)
    passed = sum(1 for c in all_checks if c["passed"])

    def _section(title: str, checks: list[dict]) -> None:
        sec_pass = sum(1 for c in checks if c["passed"])
        print(f"\n{'─' * 62}")
        print(f"  {title}  ({sec_pass}/{len(checks)} passed)")
        print(f"{'─' * 62}")
        for c in checks:
            print(f"  {_tag(c['passed'])}  {c['name']}")
            print(f"         {c['detail']}")

    print()
    print("=" * 62)
    print(f"  THREE-STAGE GATE REPORT  (combos: {', '.join(combos_run)})")
    print("=" * 62)

    _section("STAGE A — Validation", a_checks)
    _section("STAGE B — Evaluation", b_checks)
    _section("STAGE C — Stress Testing", c_checks)

    print(f"\n{'─' * 62}")
    print("  KEY METRICS")
    print(f"{'─' * 62}")
    for key in ["rmse_position", "rmse_y", "rmse_v", "max_position_error",
                "total_control_effort", "avg_solve_time_s", "max_solve_time_s",
                "optimal_solve_pct", "mean_sigma_y_norm"]:
        val = metrics.get(key, "N/A")
        if isinstance(val, float):
            print(f"  {key:30s}  {val:.6f}")
        else:
            print(f"  {key:30s}  {val}")

    a_pass = all(c["passed"] for c in a_checks)
    b_pass = all(c["passed"] for c in b_checks)
    c_pass = all(c["passed"] for c in c_checks)
    gate_pass = a_pass and b_pass and c_pass

    print(f"\n{'=' * 62}")
    print(f"  VERDICT:  {_tag(gate_pass)}  ({passed}/{total} checks passed)")
    print(f"    Stage A (Validation):    {_tag(a_pass)}")
    print(f"    Stage B (Evaluation):    {_tag(b_pass)}")
    print(f"    Stage C (Stress Test):   {_tag(c_pass)}")
    print(f"  Wall time: {wall_time:.1f}s")
    print("=" * 62)

    report = {
        "combos": combos_run,
        "gate_passed": bool(gate_pass),
        "total_checks": total,
        "checks_passed": passed,
        "stage_a_passed": bool(a_pass),
        "stage_b_passed": bool(b_pass),
        "stage_c_passed": bool(c_pass),
        "wall_time_s": round(wall_time, 1),
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
        "checks": [{"name": c["name"], "passed": bool(c["passed"]), "detail": c["detail"]}
                    for c in all_checks],
    }
    return report


# ====================================================================
#  Entry point
# ====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="DeePC three-stage gate")
    parser.add_argument(
        "--combo", type=str, default="all",
        choices=["base", "na", "oh", "na+oh", "all"],
        help="Feature combo to test (default: all)",
    )
    args = parser.parse_args()

    if args.combo == "all":
        combos_to_run = list(COMBOS.keys())
    else:
        combos_to_run = [args.combo]

    t_wall = time.perf_counter()
    config = DeePCConfig()

    print("Collecting offline data for validation...")
    u_data, y_data = collect_data(config, seed=42)
    print(f"  Data: u {u_data.shape}, y {y_data.shape}\n")

    # Stage A: structural checks (run once, feature-independent)
    print("Running Stage A — Validation...")
    a_checks = stage_a(config, u_data, y_data)

    # Stage B: per-combo evaluation
    all_b_checks: list[dict] = []
    last_metrics: dict = {}
    for combo_name in combos_to_run:
        features = COMBOS[combo_name]
        print(f"\nRunning Stage B — Evaluation [{combo_name}]...")
        b_checks, metrics = stage_b(config, features, combo_name)
        all_b_checks.extend(b_checks)
        last_metrics = metrics

    # Stage C: per-combo stress testing
    all_c_checks: list[dict] = []
    for combo_name in combos_to_run:
        features = COMBOS[combo_name]
        print(f"\nRunning Stage C — Stress Testing [{combo_name}]...")
        c_checks = stage_c(features, combo_name)
        all_c_checks.extend(c_checks)

    wall_time = time.perf_counter() - t_wall

    report = print_report(a_checks, all_b_checks, all_c_checks,
                          last_metrics, wall_time, combos_to_run)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "gate_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    if not report["gate_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
