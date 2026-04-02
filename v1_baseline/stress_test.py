"""Stress tests for v1_baseline DeePC controller (bicycle model plant).

Tests the controller under challenging conditions:
  1. High measurement noise
  2. Aggressive reference trajectory (high frequency/amplitude)
  3. Reduced data (poor excitation / small dataset)
  4. Actuator saturation (tight input constraints)
  5. Nonlinear regime (large steering angles, high speed)
  6. Step reference change (sudden lateral offset)
  7. Disturbance rejection (external velocity perturbation)
  8. Long horizon stress (extended simulation)

Each test logs PASS/FAIL with diagnostics and generates plots.

Usage (from repo root):
    uv run python -m v1_baseline.stress_test
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.parameters import DeePCConfig
from data.data_generation import collect_data
from deepc.deepc_controller import DeePCController
from plants.bicycle_model import BicycleModel

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("stress_test")
logger.setLevel(logging.INFO)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

plot_data: dict[str, dict] = {}


def _run_closed_loop(
    config: DeePCConfig,
    y_ref_full: np.ndarray,
    seed: int = 42,
    disturbance_fn=None,
) -> dict:
    """Run a closed-loop DeePC simulation and return results."""
    u_data, y_data = collect_data(config, seed=seed)
    controller = DeePCController(config, u_data, y_data)

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
    statuses: list[str] = []
    solve_times: list[float] = []

    for k in range(config.sim_steps):
        step_idx = k + config.Tini
        u_ini = np.array(u_buffer[-config.Tini :])
        y_ini = np.array(y_buffer[-config.Tini :])
        y_ref_horizon = y_ref_full[step_idx : step_idx + config.N]

        t0 = time.perf_counter()
        u_opt, info = controller.solve(u_ini, y_ini, y_ref_horizon)
        solve_times.append(time.perf_counter() - t0)

        # Apply optional disturbance
        if disturbance_fn is not None:
            u_opt = disturbance_fn(k, u_opt, sim)

        y_new = sim.step(u_opt)
        u_buffer.append(u_opt)
        y_buffer.append(y_new)
        u_history.append(u_opt)
        y_history.append(y_new)
        statuses.append(info["status"])

    total = config.Tini + config.sim_steps
    return {
        "y_history": np.array(y_history),
        "u_history": np.array(u_history),
        "y_ref_history": y_ref_full[:total],
        "statuses": statuses,
        "solve_times": solve_times,
    }


def _generate_ref(config: DeePCConfig) -> np.ndarray:
    """Generate sinusoidal reference trajectory."""
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


def test_high_noise() -> bool:
    """Run with 10x measurement noise during data collection."""
    logger.info("--- Test 1: High measurement noise (10x) ---")
    config = DeePCConfig(noise_std_output=0.1)
    y_ref = _generate_ref(config)
    results = _run_closed_loop(config, y_ref)

    rmse = _compute_rmse_position(results)
    optimal_pct = (
        100.0
        * sum(1 for s in results["statuses"] if "optimal" in s)
        / len(results["statuses"])
    )

    ok = rmse < 5.0 and optimal_pct > 50.0

    plot_data["high_noise"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "Test 1: High Noise (10x)",
    }

    logger.info("  RMSE_pos=%.4f m, optimal=%.0f%%", rmse, optimal_pct)
    logger.info("  %s", PASS if ok else FAIL)
    return ok


def test_aggressive_reference() -> bool:
    """High frequency, large amplitude reference."""
    logger.info("--- Test 2: Aggressive reference trajectory ---")
    config = DeePCConfig(ref_amplitude=10.0, ref_frequency=0.1)
    y_ref = _generate_ref(config)
    results = _run_closed_loop(config, y_ref)

    rmse = _compute_rmse_position(results)
    optimal_pct = (
        100.0
        * sum(1 for s in results["statuses"] if "optimal" in s)
        / len(results["statuses"])
    )

    # Aggressive ref is hard — just check it doesn't diverge
    ok = rmse < 20.0 and optimal_pct > 30.0

    plot_data["aggressive_ref"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "Test 2: Aggressive Reference",
    }

    logger.info("  RMSE_pos=%.4f m, optimal=%.0f%%", rmse, optimal_pct)
    logger.info("  %s", PASS if ok else FAIL)
    return ok


def test_reduced_data() -> bool:
    """Run with minimal dataset (T_data = 50)."""
    logger.info("--- Test 3: Reduced dataset (T_data=50) ---")
    config = DeePCConfig(T_data=50, sim_steps=50)
    y_ref = _generate_ref(config)
    results = _run_closed_loop(config, y_ref)

    rmse = _compute_rmse_position(results)
    optimal_pct = (
        100.0
        * sum(1 for s in results["statuses"] if "optimal" in s)
        / len(results["statuses"])
    )

    # With reduced data, performance degrades — just check stability
    ok = rmse < 15.0 and optimal_pct > 20.0

    plot_data["reduced_data"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "Test 3: Reduced Data (T=50)",
    }

    logger.info("  RMSE_pos=%.4f m, optimal=%.0f%%", rmse, optimal_pct)
    logger.info("  %s", PASS if ok else FAIL)
    return ok


def test_tight_constraints() -> bool:
    """Run with very tight actuator limits."""
    logger.info("--- Test 4: Tight input constraints ---")
    config = DeePCConfig(delta_max=0.1, a_max=1.0, a_min=-1.0)
    y_ref = _generate_ref(config)
    results = _run_closed_loop(config, y_ref)

    rmse = _compute_rmse_position(results)
    u_hist = results["u_history"]

    # Verify constraints are respected
    steering_ok = np.all(np.abs(u_hist[:, 0]) <= config.delta_max + 1e-6)
    accel_ok = np.all(u_hist[:, 1] <= config.a_max + 1e-6)
    brake_ok = np.all(u_hist[:, 1] >= config.a_min - 1e-6)

    ok = steering_ok and accel_ok and brake_ok

    plot_data["tight_constraints"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "Test 4: Tight Constraints",
    }

    logger.info("  RMSE_pos=%.4f m, constraints_ok=%s", rmse, ok)
    logger.info("  %s", PASS if ok else FAIL)
    return ok


def test_nonlinear_regime() -> bool:
    """High speed with large steering — deep nonlinear territory."""
    logger.info("--- Test 5: Nonlinear regime (high speed) ---")
    config = DeePCConfig(v_ref=10.0, ref_amplitude=8.0)
    y_ref = _generate_ref(config)
    results = _run_closed_loop(config, y_ref)

    rmse = _compute_rmse_position(results)
    optimal_pct = (
        100.0
        * sum(1 for s in results["statuses"] if "optimal" in s)
        / len(results["statuses"])
    )

    ok = rmse < 20.0 and optimal_pct > 30.0

    plot_data["nonlinear"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "Test 5: Nonlinear Regime",
    }

    logger.info("  RMSE_pos=%.4f m, optimal=%.0f%%", rmse, optimal_pct)
    logger.info("  %s", PASS if ok else FAIL)
    return ok


def test_step_reference() -> bool:
    """Sudden lateral offset at mid-simulation."""
    logger.info("--- Test 6: Step reference change ---")
    config = DeePCConfig(ref_amplitude=0.0)  # straight line
    total = config.Tini + config.sim_steps + config.N
    y_ref = np.zeros((total, config.p))
    for k in range(total):
        t = k * config.Ts
        y_ref[k, 0] = config.v_ref * t
        y_ref[k, 2] = config.v_ref
        # Step change at midpoint
        if k >= total // 2:
            y_ref[k, 1] = 3.0

    results = _run_closed_loop(config, y_ref)
    rmse = _compute_rmse_position(results)

    # Check that it converges to the offset in the second half
    y_hist = results["y_history"]
    second_half = y_hist[len(y_hist) // 2 + 20 :]  # skip transient
    if len(second_half) > 0:
        final_y_error = abs(np.mean(second_half[:, 1]) - 3.0)
    else:
        final_y_error = float("inf")

    ok = final_y_error < 2.0

    plot_data["step_ref"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "Test 6: Step Reference",
    }

    logger.info("  RMSE_pos=%.4f m, final_y_error=%.4f m", rmse, final_y_error)
    logger.info("  %s", PASS if ok else FAIL)
    return ok


def test_disturbance_rejection() -> bool:
    """Apply velocity disturbances during simulation."""
    logger.info("--- Test 7: Disturbance rejection ---")
    config = DeePCConfig()
    y_ref = _generate_ref(config)

    def disturbance(k, u_opt, sim):
        if 40 <= k <= 60:
            sim.state[3] += 0.5  # velocity kick
        return u_opt

    results = _run_closed_loop(config, y_ref, disturbance_fn=disturbance)
    rmse = _compute_rmse_position(results)
    optimal_pct = (
        100.0
        * sum(1 for s in results["statuses"] if "optimal" in s)
        / len(results["statuses"])
    )

    ok = rmse < 5.0

    plot_data["disturbance"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "Test 7: Disturbance Rejection",
    }

    logger.info("  RMSE_pos=%.4f m, optimal=%.0f%%", rmse, optimal_pct)
    logger.info("  %s", PASS if ok else FAIL)
    return ok


def test_long_horizon() -> bool:
    """Extended simulation (500 steps)."""
    logger.info("--- Test 8: Long horizon (500 steps) ---")
    config = DeePCConfig(sim_steps=500, T_data=400)
    y_ref = _generate_ref(config)
    results = _run_closed_loop(config, y_ref)

    rmse = _compute_rmse_position(results)
    avg_solve = float(np.mean(results["solve_times"]))
    max_solve = float(np.max(results["solve_times"]))

    ok = rmse < 3.0 and max_solve < 2.0

    plot_data["long_horizon"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "Test 8: Long Horizon (500 steps)",
    }

    logger.info(
        "  RMSE_pos=%.4f m, avg_solve=%.3fs, max_solve=%.3fs",
        rmse,
        avg_solve,
        max_solve,
    )
    logger.info("  %s", PASS if ok else FAIL)
    return ok


def generate_plots(results_dir: pathlib.Path) -> None:
    """Generate stress test visualization plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_tests = len(plot_data)
    if n_tests == 0:
        return

    cols = 3
    rows = (n_tests + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
    fig.suptitle("v1_baseline — Stress Test Results", fontsize=16, fontweight="bold")
    axes_flat = axes.flatten() if n_tests > 1 else [axes]

    for i, (key, d) in enumerate(plot_data.items()):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        y_hist = d["y_hist"]
        y_ref = d["y_ref"]
        n = min(len(y_hist), len(y_ref))
        ax.plot(y_ref[:n, 0], y_ref[:n, 1], "r--", linewidth=1, label="Reference")
        ax.plot(y_hist[:n, 0], y_hist[:n, 1], "b-", linewidth=0.8, label="Actual")
        ax.set_title(d["title"], fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = results_dir / "v1_baseline_stress_tests.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Stress test plots saved to %s", save_path)


def main() -> None:
    """Run all stress tests and generate plots."""
    print("=" * 62)
    print("  v1_baseline STRESS TESTS")
    print("=" * 62)

    tests = [
        ("High measurement noise", test_high_noise),
        ("Aggressive reference", test_aggressive_reference),
        ("Reduced dataset", test_reduced_data),
        ("Tight input constraints", test_tight_constraints),
        ("Nonlinear regime", test_nonlinear_regime),
        ("Step reference change", test_step_reference),
        ("Disturbance rejection", test_disturbance_rejection),
        ("Long horizon", test_long_horizon),
    ]

    results = []
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as e:
            logger.error("  %s CRASHED: %s", name, e)
            ok = False
        results.append((name, ok))

    # Generate plots
    results_dir = PROJECT_ROOT.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    generate_plots(results_dir)

    print()
    print("=" * 62)
    print("  STRESS TEST SUMMARY")
    print("=" * 62)
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {n_pass}/{len(results)} tests passed")
    print("=" * 62)

    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
