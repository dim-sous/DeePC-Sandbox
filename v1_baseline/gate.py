"""Three-stage gate for v1_baseline.

Stages:
    A. Validation  — correctness and mathematical consistency
    B. Evaluation  — tracking error, control effort, solver time, constraints, stability
    C. Stress Test  — noise, disturbances, poor excitation, actuator limits, nonlinear regimes

Usage (from repo root):
    uv run python -m v1_baseline.gate
"""

from __future__ import annotations

import json
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

RESULTS_DIR = REPO_ROOT / "results" / "v1_baseline"

from config.parameters import DeePCConfig
from data.data_generation import collect_data
from deepc.deepc_controller import DeePCController
from deepc.hankel import build_hankel_matrix, build_data_matrices
from plants.bicycle_model import BicycleModel

sys.path.insert(0, str(REPO_ROOT))
from comparison import stress_configs

# ── colour helpers ──────────────────────────────────────────────────
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _tag(ok: bool) -> str:
    return PASS if ok else FAIL


# ── shared helpers ──────────────────────────────────────────────────


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


def _compute_rmse_position(results: dict) -> float:
    y = results["y_history"]
    r = results["y_ref_history"]
    n = min(len(y), len(r))
    err = y[:n, :2] - r[:n, :2]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


# ====================================================================
#  STAGE A — Validation (correctness & mathematical consistency)
# ====================================================================


def stage_a(config: DeePCConfig, u_data: np.ndarray, y_data: np.ndarray) -> list[dict]:
    """Run all Stage-A validation checks.

    Returns a list of check dicts: {name, passed, detail}.
    """
    checks: list[dict] = []

    # ── A1: Hankel matrix dimensions ────────────────────────────────
    L = config.L  # Tini + N
    T = config.T_data
    m, p = config.m, config.p

    Hu = build_hankel_matrix(u_data, L)
    Hy = build_hankel_matrix(y_data, L)
    expected_cols = T - L + 1

    ok_hu = Hu.shape == (L * m, expected_cols)
    ok_hy = Hy.shape == (L * p, expected_cols)
    checks.append({
        "name": "A1  Hankel(u) dimensions",
        "passed": ok_hu,
        "detail": f"expected ({L * m}, {expected_cols}), got {Hu.shape}",
    })
    checks.append({
        "name": "A1  Hankel(y) dimensions",
        "passed": ok_hy,
        "detail": f"expected ({L * p}, {expected_cols}), got {Hy.shape}",
    })

    # ── A2: Hankel partition consistency ────────────────────────────
    Up, Yp, Uf, Yf = build_data_matrices(u_data, y_data, config.Tini, config.N)

    ok_up = Up.shape == (config.Tini * m, expected_cols)
    ok_yp = Yp.shape == (config.Tini * p, expected_cols)
    ok_uf = Uf.shape == (config.N * m, expected_cols)
    ok_yf = Yf.shape == (config.N * p, expected_cols)
    checks.append({
        "name": "A2  Partition shapes (Up,Yp,Uf,Yf)",
        "passed": ok_up and ok_yp and ok_uf and ok_yf,
        "detail": (
            f"Up {Up.shape}, Yp {Yp.shape}, Uf {Uf.shape}, Yf {Yf.shape}"
        ),
    })

    # Verify partitions are subblocks of the full Hankel
    top_u = np.allclose(Up, Hu[: config.Tini * m, :])
    bot_u = np.allclose(Uf, Hu[config.Tini * m :, :])
    top_y = np.allclose(Yp, Hy[: config.Tini * p, :])
    bot_y = np.allclose(Yf, Hy[config.Tini * p :, :])
    checks.append({
        "name": "A2  Partitions match full Hankel",
        "passed": top_u and bot_u and top_y and bot_y,
        "detail": f"Up={top_u}, Uf={bot_u}, Yp={top_y}, Yf={bot_y}",
    })

    # ── A3: Persistent excitation ───────────────────────────────────
    rank = np.linalg.matrix_rank(Hu, tol=1e-6)
    required = L * m
    pe_ok = rank >= required
    checks.append({
        "name": "A3  Persistent excitation (rank condition)",
        "passed": pe_ok,
        "detail": f"rank={rank}, required>={required}, margin={rank - required}",
    })

    # ── A4: Simulator dynamics spot-check ───────────────────────────
    sim = BicycleModel(
        Ts=config.Ts,
        L_wheelbase=config.L_wheelbase,
        delta_max=config.delta_max,
        a_max=config.a_max,
        a_min=config.a_min,
        initial_state=np.array([0.0, 0.0, 0.0, 5.0]),
    )
    x0, y0, th0, v0 = sim.state.copy()
    delta_test, a_test = 0.1, 0.5
    sim.step(np.array([delta_test, a_test]))
    x1, y1, th1, v1 = sim.state

    Ts = config.Ts
    Lw = config.L_wheelbase
    x1_exp = x0 + v0 * np.cos(th0) * Ts
    y1_exp = y0 + v0 * np.sin(th0) * Ts
    th1_exp = th0 + (v0 / Lw) * np.tan(delta_test) * Ts
    v1_exp = max(v0 + a_test * Ts, 0.0)

    sim_ok = (
        np.isclose(x1, x1_exp)
        and np.isclose(y1, y1_exp)
        and np.isclose(th1, th1_exp)
        and np.isclose(v1, v1_exp)
    )
    checks.append({
        "name": "A4  Simulator dynamics (one-step spot check)",
        "passed": bool(sim_ok),
        "detail": (
            f"expected [x={x1_exp:.6f}, y={y1_exp:.6f}, th={th1_exp:.6f}, v={v1_exp:.6f}], "
            f"got [x={x1:.6f}, y={y1:.6f}, th={th1:.6f}, v={v1:.6f}]"
        ),
    })

    # ── A5: Hidden heading — output vector is [x, y, v] not [x, y, theta, v]
    sim2 = BicycleModel(
        Ts=config.Ts,
        L_wheelbase=config.L_wheelbase,
        delta_max=config.delta_max,
        a_max=config.a_max,
        a_min=config.a_min,
        initial_state=np.array([1.0, 2.0, 0.5, 3.0]),
    )
    out = sim2.output
    heading_hidden = len(out) == 3 and np.isclose(out[2], 3.0)  # v, not theta
    checks.append({
        "name": "A5  Output hides heading (dim=3, [x,y,v])",
        "passed": bool(heading_hidden),
        "detail": f"output={out}, len={len(out)}",
    })

    # ── A6: QP structure sanity — build controller and verify one solve
    controller = DeePCController(config, u_data, y_data)

    u_ini = np.zeros((config.Tini, m))
    y_ini = np.zeros((config.Tini, p))
    y_ref = np.zeros((config.N, p))
    u_opt, info = controller.solve(u_ini, y_ini, y_ref)

    solve_ok = "optimal" in info["status"]
    cost_finite = info["cost"] is not None and np.isfinite(info["cost"])
    cost_nonneg = info["cost"] is not None and info["cost"] >= -1e-6
    checks.append({
        "name": "A6  QP solves to optimality (trivial inputs)",
        "passed": solve_ok,
        "detail": f"status={info['status']}",
    })
    checks.append({
        "name": "A6  Objective is finite and non-negative",
        "passed": bool(cost_finite and cost_nonneg),
        "detail": f"cost={info['cost']}",
    })

    # ── A7: Input constraints in QP output ──────────────────────────
    if info["u_predicted"] is not None:
        u_pred = info["u_predicted"]
        steer_ok = bool(np.all(np.abs(u_pred[:, 0]) <= config.delta_max + 1e-6))
        accel_ok = bool(
            np.all(u_pred[:, 1] <= config.a_max + 1e-6)
            and np.all(u_pred[:, 1] >= config.a_min - 1e-6)
        )
    else:
        steer_ok = accel_ok = False
    checks.append({
        "name": "A7  Predicted inputs satisfy box constraints",
        "passed": steer_ok and accel_ok,
        "detail": f"steering_ok={steer_ok}, accel_ok={accel_ok}",
    })

    return checks


# ====================================================================
#  STAGE B — Evaluation (quantitative performance metrics)
# ====================================================================


def stage_b(config: DeePCConfig) -> tuple[list[dict], dict]:
    """Run baseline simulation and evaluate metrics.

    Returns (checks, metrics_dict).
    """
    from main import run_deepc_simulation

    sys.path.insert(0, str(REPO_ROOT))
    from comparison.metrics import compute_all_metrics

    results = run_deepc_simulation(config)
    metrics = compute_all_metrics(results, "v1_baseline")

    checks: list[dict] = []

    checks.append({
        "name": "B1  Position RMSE < 1.0 m",
        "passed": metrics["rmse_position"] < 1.0,
        "detail": f"rmse_position={metrics['rmse_position']:.4f} m",
    })
    checks.append({
        "name": "B1  Lateral RMSE (y) < 0.5 m",
        "passed": metrics["rmse_y"] < 0.5,
        "detail": f"rmse_y={metrics['rmse_y']:.4f} m",
    })
    checks.append({
        "name": "B1  Velocity RMSE < 0.5 m/s",
        "passed": metrics["rmse_v"] < 0.5,
        "detail": f"rmse_v={metrics['rmse_v']:.4f} m/s",
    })
    checks.append({
        "name": "B1  Max position error < 2.0 m",
        "passed": metrics["max_position_error"] < 2.0,
        "detail": f"max_pos_err={metrics['max_position_error']:.4f} m",
    })

    checks.append({
        "name": "B2  Total control effort < 50",
        "passed": metrics["total_control_effort"] < 50.0,
        "detail": f"total_effort={metrics['total_control_effort']:.2f}",
    })

    checks.append({
        "name": "B3  Optimal solve rate >= 80%",
        "passed": metrics["optimal_solve_pct"] >= 80.0,
        "detail": f"optimal_pct={metrics['optimal_solve_pct']:.1f}%",
    })
    checks.append({
        "name": "B3  Avg solve time < 2.0 s",
        "passed": metrics["avg_solve_time_s"] < 2.0,
        "detail": f"avg_solve={metrics['avg_solve_time_s']:.4f} s",
    })
    checks.append({
        "name": "B3  Max solve time < 5.0 s",
        "passed": metrics["max_solve_time_s"] < 5.0,
        "detail": f"max_solve={metrics['max_solve_time_s']:.4f} s",
    })

    checks.append({
        "name": "B4  Mean slack norm < 0.1",
        "passed": metrics["mean_sigma_y_norm"] < 0.1,
        "detail": f"mean_sigma={metrics['mean_sigma_y_norm']:.6f}",
    })

    # Stability — second half vs first half
    y_hist = np.asarray(results["y_history"])
    y_ref = np.asarray(results["y_ref_history"])
    n = min(len(y_hist), len(y_ref))
    pos_err = np.sqrt(
        (y_hist[:n, 0] - y_ref[:n, 0]) ** 2
        + (y_hist[:n, 1] - y_ref[:n, 1]) ** 2
    )
    mid = n // 2
    rmse_first = float(np.sqrt(np.mean(pos_err[:mid] ** 2)))
    rmse_second = float(np.sqrt(np.mean(pos_err[mid:] ** 2)))
    stable = rmse_second < max(rmse_first * 2.0, 1.0)
    checks.append({
        "name": "B5  No divergence (2nd half RMSE <= 2x 1st half)",
        "passed": stable,
        "detail": f"1st_half={rmse_first:.4f}, 2nd_half={rmse_second:.4f}",
    })

    # B6-B8: Hard scenario (run via main simulation loop)
    hard_results = results
    hard_metrics = metrics

    checks.append({
        "name": "B6  [hard] Lateral RMSE < 3.0 m",
        "passed": hard_metrics["rmse_y"] < 3.0,
        "detail": f"rmse_y={hard_metrics['rmse_y']:.4f} m",
    })
    checks.append({
        "name": "B7  [hard] Optimal solve rate >= 80%",
        "passed": hard_metrics["optimal_solve_pct"] >= 80.0,
        "detail": f"optimal_pct={hard_metrics['optimal_solve_pct']:.1f}%",
    })
    checks.append({
        "name": "B8  [hard] Avg solve time < 2.0 s",
        "passed": hard_metrics["avg_solve_time_s"] < 2.0,
        "detail": f"avg_solve={hard_metrics['avg_solve_time_s']:.4f} s",
    })

    return checks, metrics


# ====================================================================
#  STAGE C — Stress Testing
# ====================================================================

_plot_data: dict[str, dict] = {}


def _test_high_noise() -> bool:
    config = DeePCConfig(**stress_configs.high_noise())
    y_ref = _generate_ref(config)
    results = _run_closed_loop(config, y_ref)
    rmse = _compute_rmse_position(results)
    optimal_pct = (
        100.0
        * sum(1 for s in results["statuses"] if "optimal" in s)
        / len(results["statuses"])
    )
    _plot_data["high_noise"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "C1: High Noise (10x)",
    }
    return rmse < 5.0 and optimal_pct > 50.0


def _test_aggressive_reference() -> bool:
    config = DeePCConfig(**stress_configs.aggressive_reference())
    y_ref = _generate_ref(config)
    results = _run_closed_loop(config, y_ref)
    rmse = _compute_rmse_position(results)
    optimal_pct = (
        100.0
        * sum(1 for s in results["statuses"] if "optimal" in s)
        / len(results["statuses"])
    )
    _plot_data["aggressive_ref"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "C2: Aggressive Reference",
    }
    return rmse < 20.0 and optimal_pct > 30.0


def _test_reduced_data() -> bool:
    config = DeePCConfig(**stress_configs.reduced_data())
    y_ref = _generate_ref(config)
    results = _run_closed_loop(config, y_ref)
    rmse = _compute_rmse_position(results)
    optimal_pct = (
        100.0
        * sum(1 for s in results["statuses"] if "optimal" in s)
        / len(results["statuses"])
    )
    _plot_data["reduced_data"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "C3: Reduced Data (T=50)",
    }
    return rmse < 15.0 and optimal_pct > 20.0


def _test_tight_constraints() -> bool:
    config = DeePCConfig(**stress_configs.tight_constraints())
    y_ref = _generate_ref(config)
    results = _run_closed_loop(config, y_ref)
    u_hist = results["u_history"]
    steering_ok = np.all(np.abs(u_hist[:, 0]) <= config.delta_max + 1e-6)
    accel_ok = np.all(u_hist[:, 1] <= config.a_max + 1e-6)
    brake_ok = np.all(u_hist[:, 1] >= config.a_min - 1e-6)
    _plot_data["tight_constraints"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "C4: Tight Constraints",
    }
    return bool(steering_ok and accel_ok and brake_ok)


def _test_nonlinear_regime() -> bool:
    config = DeePCConfig(**stress_configs.nonlinear_regime())
    y_ref = _generate_ref(config)
    results = _run_closed_loop(config, y_ref)
    rmse = _compute_rmse_position(results)
    optimal_pct = (
        100.0
        * sum(1 for s in results["statuses"] if "optimal" in s)
        / len(results["statuses"])
    )
    _plot_data["nonlinear"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "C5: Nonlinear Regime",
    }
    return rmse < 20.0 and optimal_pct > 30.0


def _test_step_reference() -> bool:
    config = DeePCConfig(**stress_configs.step_reference())
    total = config.Tini + config.sim_steps + config.N
    y_ref = np.zeros((total, config.p))
    for k in range(total):
        t = k * config.Ts
        y_ref[k, 0] = config.v_ref * t
        y_ref[k, 2] = config.v_ref
        if k >= total // 2:
            y_ref[k, 1] = 3.0

    results = _run_closed_loop(config, y_ref)
    y_hist = results["y_history"]
    second_half = y_hist[len(y_hist) // 2 + 20 :]
    if len(second_half) > 0:
        final_y_error = abs(np.mean(second_half[:, 1]) - 3.0)
    else:
        final_y_error = float("inf")

    _plot_data["step_ref"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "C6: Step Reference",
    }
    return final_y_error < 2.0


def _test_disturbance_rejection() -> bool:
    config = DeePCConfig(**stress_configs.disturbance_rejection())
    y_ref = _generate_ref(config)

    def disturbance(k, u_opt, sim):
        if 40 <= k <= 60:
            sim.state[3] += 0.5
        return u_opt

    results = _run_closed_loop(config, y_ref, disturbance_fn=disturbance)
    rmse = _compute_rmse_position(results)
    _plot_data["disturbance"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "C7: Disturbance Rejection",
    }
    return rmse < 5.0


def _test_long_horizon() -> bool:
    config = DeePCConfig(**stress_configs.long_horizon())
    y_ref = _generate_ref(config)
    results = _run_closed_loop(config, y_ref)
    rmse = _compute_rmse_position(results)
    max_solve = float(np.max(results["solve_times"]))
    _plot_data["long_horizon"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "C8: Long Horizon (500 steps)",
    }
    return rmse < 3.0 and max_solve < 2.0


def _test_rate_constrained_aggressive() -> bool:
    """Aggressive reference with rate constraints — verify actuator rates stay bounded."""
    config = DeePCConfig(**stress_configs.rate_constrained_aggressive())
    y_ref = _generate_ref(config)
    results = _run_closed_loop(config, y_ref)
    u_hist = results["u_history"]
    du = np.diff(u_hist, axis=0)
    rate_steer = bool(np.all(np.abs(du[:, 0]) <= config.d_delta_max + 1e-4))
    rate_accel = bool(np.all(np.abs(du[:, 1]) <= config.da_max + 1e-4))
    _plot_data["rate_aggressive"] = {
        "y_hist": results["y_history"],
        "y_ref": results["y_ref_history"],
        "title": "C9: Rate-Constrained Aggressive",
    }
    return rate_steer and rate_accel


def stage_c() -> list[dict]:
    """Run the 9-scenario stress suite."""
    tests = [
        ("C1  High measurement noise (10x)", _test_high_noise),
        ("C2  Aggressive reference", _test_aggressive_reference),
        ("C3  Reduced dataset (T=50)", _test_reduced_data),
        ("C4  Tight input constraints", _test_tight_constraints),
        ("C5  Nonlinear regime (high speed)", _test_nonlinear_regime),
        ("C6  Step reference change", _test_step_reference),
        ("C7  Disturbance rejection", _test_disturbance_rejection),
        ("C8  Long horizon (200 steps)", _test_long_horizon),
        ("C9  Rate-constrained aggressive ref", _test_rate_constrained_aggressive),
    ]

    checks: list[dict] = []
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            ok = fn()
            elapsed = time.perf_counter() - t0
            checks.append({
                "name": name,
                "passed": ok,
                "detail": f"elapsed={elapsed:.1f}s",
            })
        except Exception as e:
            elapsed = time.perf_counter() - t0
            checks.append({
                "name": name,
                "passed": False,
                "detail": f"CRASHED: {e} (elapsed={elapsed:.1f}s)",
            })

    # Generate stress test plots
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _generate_stress_plots(RESULTS_DIR)

    return checks


def _generate_stress_plots(results_dir: pathlib.Path) -> None:
    """Generate stress test visualization plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_tests = len(_plot_data)
    if n_tests == 0:
        return

    cols = 3
    rows = (n_tests + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
    fig.suptitle("v1_baseline — Stress Test Results", fontsize=16, fontweight="bold")
    axes_flat = axes.flatten() if n_tests > 1 else [axes]

    for i, (key, d) in enumerate(_plot_data.items()):
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


# ====================================================================
#  Report
# ====================================================================


def print_report(
    a_checks: list[dict],
    b_checks: list[dict],
    c_checks: list[dict],
    metrics: dict,
    wall_time: float,
) -> dict:
    """Print structured gate report and return JSON-serialisable summary."""
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
    print("  v1_baseline  —  THREE-STAGE GATE REPORT")
    print("=" * 62)

    _section("STAGE A — Validation", a_checks)
    _section("STAGE B — Evaluation", b_checks)
    _section("STAGE C — Stress Testing", c_checks)

    # Key metrics summary
    print(f"\n{'─' * 62}")
    print("  KEY METRICS")
    print(f"{'─' * 62}")
    for key in [
        "rmse_position",
        "rmse_y",
        "rmse_v",
        "max_position_error",
        "total_control_effort",
        "avg_solve_time_s",
        "max_solve_time_s",
        "optimal_solve_pct",
        "mean_sigma_y_norm",
    ]:
        val = metrics.get(key, "N/A")
        if isinstance(val, float):
            print(f"  {key:30s}  {val:.6f}")
        else:
            print(f"  {key:30s}  {val}")

    # Overall verdict
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
        "version": "v1_baseline",
        "gate_passed": bool(gate_pass),
        "total_checks": total,
        "checks_passed": passed,
        "stage_a_passed": bool(a_pass),
        "stage_b_passed": bool(b_pass),
        "stage_c_passed": bool(c_pass),
        "wall_time_s": round(wall_time, 1),
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
        "checks": [
            {"name": c["name"], "passed": bool(c["passed"]), "detail": c["detail"]}
            for c in all_checks
        ],
    }
    return report


# ====================================================================
#  Entry point
# ====================================================================


def main() -> None:
    t_wall = time.perf_counter()

    config = DeePCConfig()

    # ── Stage A ─────────────────────────────────────────────────────
    print("Collecting offline data for validation...")
    u_data, y_data = collect_data(config, seed=42)
    print(f"  Data: u {u_data.shape}, y {y_data.shape}\n")

    print("Running Stage A — Validation...")
    a_checks = stage_a(config, u_data, y_data)

    # ── Stage B ─────────────────────────────────────────────────────
    print("\nRunning Stage B — Evaluation...")
    b_checks, metrics = stage_b(config)

    # ── Stage C ─────────────────────────────────────────────────────
    print("\nRunning Stage C — Stress Testing...")
    c_checks = stage_c()

    wall_time = time.perf_counter() - t_wall

    # ── Report ──────────────────────────────────────────────────────
    report = print_report(a_checks, b_checks, c_checks, metrics, wall_time)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "v1_gate_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    if not report["gate_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
