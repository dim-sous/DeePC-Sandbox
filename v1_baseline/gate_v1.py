"""Three-stage gate for v1_baseline.

Stages:
    A. Validation  — correctness and mathematical consistency
    B. Evaluation  — tracking error, control effort, solver time, constraints, stability
    C. Stress Test  — noise, disturbances, poor excitation, actuator limits, nonlinear regimes

Usage (from repo root):
    uv run python -m v1_baseline.gate_v1
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
RESULTS_DIR = REPO_ROOT / "results"

from config.parameters import DeePCConfig
from data.data_generation import collect_data
from deepc.deepc_controller import DeePCController
from deepc.hankel import build_hankel_matrix, build_data_matrices
from simulation.vehicle_simulator import VehicleSimulator

# ── colour helpers ──────────────────────────────────────────────────
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def _tag(ok: bool) -> str:
    return PASS if ok else FAIL


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
    #  Run one step with known inputs and verify the kinematic bicycle
    #  equations are satisfied.
    sim = VehicleSimulator(config, initial_state=np.array([0.0, 0.0, 0.0, 5.0]))
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
            f"expected [x={x1_exp:.6f}, y={y1_exp:.6f}, θ={th1_exp:.6f}, v={v1_exp:.6f}], "
            f"got [x={x1:.6f}, y={y1:.6f}, θ={th1:.6f}, v={v1:.6f}]"
        ),
    })

    # ── A5: Hidden heading — output vector is [x, y, v] not [x, y, θ, v] ─
    sim2 = VehicleSimulator(config, initial_state=np.array([1.0, 2.0, 0.5, 3.0]))
    out = sim2.output
    heading_hidden = len(out) == 3 and np.isclose(out[2], 3.0)  # v, not θ
    checks.append({
        "name": "A5  Output hides heading (dim=3, [x,y,v])",
        "passed": bool(heading_hidden),
        "detail": f"output={out}, len={len(out)}",
    })

    # ── A6: QP structure sanity — build controller and verify one solve ─
    controller = DeePCController(config, u_data, y_data)

    # Feed trivial past and reference
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

    # Tracking accuracy
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

    # Control effort
    checks.append({
        "name": "B2  Total control effort < 50",
        "passed": metrics["total_control_effort"] < 50.0,
        "detail": f"total_effort={metrics['total_control_effort']:.2f}",
    })

    # Solver performance
    checks.append({
        "name": "B3  Optimal solve rate = 100%",
        "passed": metrics["optimal_solve_pct"] == 100.0,
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

    # Constraint satisfaction (slack)
    checks.append({
        "name": "B4  Mean slack norm < 0.1",
        "passed": metrics["mean_sigma_y_norm"] < 0.1,
        "detail": f"mean_sigma={metrics['mean_sigma_y_norm']:.6f}",
    })

    # Stability — check no divergence in second half vs first half
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
    # Second half should not be drastically worse (allowing 2x for warm-up transient)
    stable = rmse_second < max(rmse_first * 2.0, 1.0)
    checks.append({
        "name": "B5  No divergence (2nd half RMSE ≤ 2× 1st half)",
        "passed": stable,
        "detail": f"1st_half={rmse_first:.4f}, 2nd_half={rmse_second:.4f}",
    })

    return checks, metrics


# ====================================================================
#  STAGE C — Stress Testing
# ====================================================================


def stage_c() -> list[dict]:
    """Run the 8-scenario stress suite.

    Returns a list of check dicts.
    """
    from stress_test import (
        test_high_noise,
        test_aggressive_reference,
        test_reduced_data,
        test_tight_constraints,
        test_nonlinear_regime,
        test_step_reference,
        test_disturbance_rejection,
        test_long_horizon,
        generate_plots,
    )

    tests = [
        ("C1  High measurement noise (10×)", test_high_noise),
        ("C2  Aggressive reference", test_aggressive_reference),
        ("C3  Reduced dataset (T=50)", test_reduced_data),
        ("C4  Tight input constraints", test_tight_constraints),
        ("C5  Nonlinear regime (high speed)", test_nonlinear_regime),
        ("C6  Step reference change", test_step_reference),
        ("C7  Disturbance rejection", test_disturbance_rejection),
        ("C8  Long horizon (500 steps)", test_long_horizon),
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
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    try:
        generate_plots(results_dir)
    except Exception:
        pass

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

    # Build JSON report
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
