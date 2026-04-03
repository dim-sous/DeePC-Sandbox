"""Standalone DeePC experiment runner.

Generates training data, builds a reference trajectory, runs closed-loop
simulation, computes metrics, and produces a self-contained HTML report
with interactive Plotly charts.

Usage (from repo root):
    uv run python run.py                          # defaults
    uv run python run.py --scenario hard --N 20
    uv run python run.py --T-data 200 --Tini 5 --lambda-g 10
    uv run python run.py --Q 10 10 1 --R 0.1 0.1
    uv run python run.py --help                   # full list
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import fields
from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from control.config import DeePCConfig
from sim.data_generation import collect_data
from sim.scenarios import get_reference
from sim.simulation import FeatureFlags, run_simulation

REPO_ROOT = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"

SCENARIOS = ["default"]

_C_REF = "#d62728"
_C_ACT = "#1f77b4"
_C_ERR = "#2ca02c"
_C_STEER = "#1f77b4"
_C_ACCEL = "#ff7f0e"
_C_BAND = "#aaaaaa"

_PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=50, r=30, t=40, b=40),
    hoverlabel=dict(font_size=11),
)


# ── Metrics ───────────────────────────────────────────────────────────

def compute_metrics(results: dict, tag: str) -> dict[str, float | str]:
    """Compute performance metrics from simulation results."""
    metrics: dict[str, float | str] = {"version": tag}

    # Absolute position tracking
    pos = np.asarray(results["pos_history"])
    ref_pos = np.asarray(results["ref_pos_history"])
    n_pos = min(len(pos), len(ref_pos))
    pos_err = pos[:n_pos] - ref_pos[:n_pos]
    metrics["rmse_x"] = float(np.sqrt(np.mean(pos_err[:, 0] ** 2)))
    metrics["rmse_y"] = float(np.sqrt(np.mean(pos_err[:, 1] ** 2)))
    metrics["rmse_position"] = float(
        np.sqrt(np.mean(pos_err[:, 0] ** 2 + pos_err[:, 1] ** 2))
    )

    # Error-based outputs [e_lat, e_heading, e_v]
    errors = np.asarray(results["y_history"])
    metrics["rmse_lateral"] = float(np.sqrt(np.mean(errors[:, 0] ** 2)))
    metrics["rmse_heading"] = float(np.sqrt(np.mean(errors[:, 1] ** 2)))
    metrics["rmse_v"] = float(np.sqrt(np.mean(errors[:, 2] ** 2)))
    solve_times = results.get("solve_times")
    if solve_times is not None and len(solve_times) > 0:
        metrics["avg_solve_time_s"] = float(np.mean(solve_times))
        metrics["max_solve_time_s"] = float(np.max(solve_times))
        metrics["total_solve_time_s"] = float(np.sum(solve_times))
    else:
        metrics["avg_solve_time_s"] = float("nan")
        metrics["max_solve_time_s"] = float("nan")
        metrics["total_solve_time_s"] = float("nan")

    statuses = results.get("statuses", [])
    if statuses:
        optimal_count = sum(1 for s in statuses if "optimal" in s)
        metrics["optimal_solve_pct"] = float(100.0 * optimal_count / len(statuses))
    else:
        metrics["optimal_solve_pct"] = float("nan")

    sigma_norms = results.get("sigma_norms")
    if sigma_norms is not None:
        norms = [s for s in sigma_norms if s is not None]
        if norms:
            metrics["mean_sigma_y_norm"] = float(np.mean(norms))
            metrics["max_sigma_y_norm"] = float(np.max(norms))
        else:
            metrics["mean_sigma_y_norm"] = float("nan")
            metrics["max_sigma_y_norm"] = float("nan")

    return metrics


# ── Plotly figures ────────────────────────────────────────────────────

def plot_training_data(
    u_data: np.ndarray, y_data: np.ndarray, config: DeePCConfig,
) -> str:
    """Return Plotly HTML div for training data."""
    T = len(u_data)
    t = (np.arange(T) * config.Ts).tolist()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Steering Excitation", "Acceleration Excitation",
            "Data Collection Trajectory", "Velocity During Collection",
        ],
        horizontal_spacing=0.08, vertical_spacing=0.12,
    )

    # Steering
    fig.add_trace(go.Scatter(
        x=t, y=u_data[:, 0].tolist(), mode="lines",
        line=dict(color=_C_STEER, width=1), name="Steering",
    ), row=1, col=1)
    for val in [config.delta_max, -config.delta_max]:
        fig.add_hline(y=val, line=dict(color=_C_BAND, dash="dot", width=1),
                      row=1, col=1)
    fig.update_xaxes(title_text="Time [s]", row=1, col=1)
    fig.update_yaxes(title_text="Steering [rad]", row=1, col=1)

    # Acceleration
    fig.add_trace(go.Scatter(
        x=t, y=u_data[:, 1].tolist(), mode="lines",
        line=dict(color=_C_ACCEL, width=1), name="Acceleration",
    ), row=1, col=2)
    for val in [config.a_max, config.a_min]:
        fig.add_hline(y=val, line=dict(color=_C_BAND, dash="dot", width=1),
                      row=1, col=2)
    fig.update_xaxes(title_text="Time [s]", row=1, col=2)
    fig.update_yaxes(title_text="Accel [m/s²]", row=1, col=2)

    # Trajectory (accumulate from increments)
    x_pos = np.cumsum(np.concatenate([[0], y_data[:, 0]]))
    y_pos = np.cumsum(np.concatenate([[0], y_data[:, 1]]))
    fig.add_trace(go.Scatter(
        x=x_pos.tolist(), y=y_pos.tolist(), mode="lines",
        line=dict(color=_C_ACT, width=1), name="Path",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[float(x_pos[0])], y=[float(y_pos[0])], mode="markers",
        marker=dict(color=_C_ACT, size=7, symbol="circle"), name="Start",
    ), row=2, col=1)
    fig.update_xaxes(title_text="x [m]", row=2, col=1)
    fig.update_yaxes(title_text="y [m]", scaleanchor="x3", scaleratio=1, row=2, col=1)

    # Velocity
    fig.add_trace(go.Scatter(
        x=t, y=y_data[:, 2].tolist(), mode="lines",
        line=dict(color=_C_ACT, width=1), name="Velocity",
    ), row=2, col=2)
    fig.update_xaxes(title_text="Time [s]", row=2, col=2)
    fig.update_yaxes(title_text="v [m/s]", row=2, col=2)

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=600,
        showlegend=False,
        title=dict(text=f"Training Data (T={T})", x=0.5, font_size=15),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _ref_actual_pair(fig, times, ref, actual, row, col, ylabel,
                     show_legend=False):
    """Add a reference/actual trace pair to a subplot."""
    fig.add_trace(go.Scatter(
        x=times, y=ref, mode="lines",
        line=dict(color=_C_REF, dash="dash", width=1.5),
        name="Reference", legendgroup="ref", showlegend=show_legend,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=times, y=actual, mode="lines",
        line=dict(color=_C_ACT, width=1.5),
        name="Actual", legendgroup="act", showlegend=show_legend,
    ), row=row, col=col)
    fig.update_xaxes(title_text="Time [s]", row=row, col=col)
    fig.update_yaxes(title_text=ylabel, row=row, col=col)


def plot_simulation_results(
    results: dict, config: DeePCConfig, tag: str,
) -> str:
    """Return Plotly HTML div for simulation results."""
    errors = np.asarray(results["y_history"])    # [e_lat, e_heading, e_v]
    u_hist = np.asarray(results["u_history"])
    times = np.asarray(results["times"])
    pos = np.asarray(results["pos_history"])      # (n+1, 2)
    ref_pos = np.asarray(results["ref_pos_history"])  # (n, 2)
    ref_path = np.asarray(results["ref_path"])    # (n, 4) [x,y,heading,v]

    n = min(len(times), len(errors), len(u_hist))
    times = times[:n].tolist()
    errors, u_hist = errors[:n], u_hist[:n]
    pos = pos[:n + 1]
    ref_pos = ref_pos[:n]
    ref_path = ref_path[:n]
    pos_times = (np.arange(n + 1) * config.Ts).tolist()

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Longitudinal Position (x)", "Lateral Position (y)",
            "Trajectory (x-y)", "Velocity",
            "Steering", "Acceleration",
        ],
        horizontal_spacing=0.08, vertical_spacing=0.08,
    )

    # Row 1: x(t), y(t) — absolute positions
    ref_times = (np.arange(len(ref_pos)) * config.Ts).tolist()
    _ref_actual_pair(fig, ref_times, ref_pos[:, 0].tolist(), pos[1:, 0].tolist(),
                     1, 1, "x [m]", show_legend=True)
    _ref_actual_pair(fig, ref_times, ref_pos[:, 1].tolist(), pos[1:, 1].tolist(),
                     1, 2, "y [m]")

    # Row 2: x/y trajectory, velocity
    fig.add_trace(go.Scatter(
        x=ref_pos[:, 0].tolist(), y=ref_pos[:, 1].tolist(), mode="lines",
        line=dict(color=_C_REF, dash="dash", width=1.5),
        name="Reference", legendgroup="ref", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=pos[:, 0].tolist(), y=pos[:, 1].tolist(), mode="lines",
        line=dict(color=_C_ACT, width=1.5),
        name="Actual", legendgroup="act", showlegend=False,
    ), row=2, col=1)
    fig.update_xaxes(title_text="x [m]", row=2, col=1)
    fig.update_yaxes(title_text="y [m]", scaleanchor="x3", scaleratio=1,
                     row=2, col=1)

    # Velocity: actual = ref_v + e_v
    ref_v = ref_path[:, 3].tolist()
    actual_v = (ref_path[:, 3] + errors[:, 2]).tolist()
    _ref_actual_pair(fig, times, ref_v, actual_v, 2, 2, "v [m/s]")
    if np.isfinite(config.y_lb[2]):
        fig.add_hline(y=config.y_lb[2],
                      line=dict(color=_C_BAND, dash="dashdot", width=1),
                      row=2, col=2)
    if np.isfinite(config.y_ub[2]):
        fig.add_hline(y=config.y_ub[2],
                      line=dict(color=_C_BAND, dash="dashdot", width=1),
                      row=2, col=2)

    # Row 3: steering, acceleration
    fig.add_trace(go.Scatter(
        x=times, y=u_hist[:, 0].tolist(), mode="lines",
        line=dict(color=_C_STEER, width=1.2),
        name="Steering", showlegend=False,
    ), row=3, col=1)
    for val in [config.delta_max, -config.delta_max]:
        fig.add_hline(y=val, line=dict(color=_C_BAND, dash="dot", width=1),
                      row=3, col=1)
    fig.update_xaxes(title_text="Time [s]", row=3, col=1)
    fig.update_yaxes(title_text="Steering [rad]", row=3, col=1)

    fig.add_trace(go.Scatter(
        x=times, y=u_hist[:, 1].tolist(), mode="lines",
        line=dict(color=_C_ACCEL, width=1.2),
        name="Acceleration", showlegend=False,
    ), row=3, col=2)
    for val in [config.a_max, config.a_min]:
        fig.add_hline(y=val, line=dict(color=_C_BAND, dash="dot", width=1),
                      row=3, col=2)
    fig.update_xaxes(title_text="Time [s]", row=3, col=2)
    fig.update_yaxes(title_text="Accel [m/s²]", row=3, col=2)

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=820,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font_size=12,
        ),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── HTML report ───────────────────────────────────────────────────────

_METRIC_GROUPS = {
    "Tracking (absolute)": [
        ("rmse_position", "Position RMSE", "m"),
        ("rmse_x", "Longitudinal RMSE", "m"),
        ("rmse_y", "Lateral RMSE", "m"),
    ],
    "Tracking (errors)": [
        ("rmse_lateral", "Lateral Error RMSE", "m"),
        ("rmse_heading", "Heading Error RMSE", "rad"),
        ("rmse_v", "Velocity Error RMSE", "m/s"),
    ],
    "Solver": [
        ("optimal_solve_pct", "Optimal Solves", "%"),
        ("avg_solve_time_s", "Avg Solve Time", "s"),
        ("max_solve_time_s", "Max Solve Time", "s"),
        ("total_solve_time_s", "Total Solve Time", "s"),
    ],
    "Constraints": [
        ("mean_sigma_y_norm", "Mean Slack ‖σ_y‖", ""),
        ("max_sigma_y_norm", "Max Slack ‖σ_y‖", ""),
    ],
}

_CONFIG_GROUPS = {
    "Horizons & Simulation": [
        "Tini", "N", "sim_duration", "Ts",
    ],
    "Data Collection": [
        "T_data", "noise_std_output", "input_amplitude_delta",
        "input_amplitude_a", "prbs_min_period",
    ],
    "Cost Weights": [
        "Q_diag", "R_diag",
    ],
    "Regularization": [
        "lambda_g", "lambda_y", "lambda_out",
        "reg_norm_g", "reg_norm_sigma_y",
    ],
    "Input Constraints": [
        "delta_max", "a_max", "a_min", "d_delta_max", "da_max",
    ],
    "Output Constraints": [
        "y_lb", "y_ub",
    ],
    "Reference": [
        "v_ref", "ref_amplitude", "ref_frequency",
    ],
    "Plant": [
        "L_wheelbase",
    ],
}

_CONFIG_LABELS = {
    "Tini": ("T_ini", "Past window"),
    "N": ("N", "Prediction horizon"),
    "sim_duration": ("T_sim", "Simulation duration [s]"),
    "Ts": ("T_s", "Sampling time [s]"),
    "T_data": ("T_data", "Training samples"),
    "noise_std_output": ("σ_noise", "Measurement noise std"),
    "input_amplitude_delta": ("Δ_amp", "Steering excitation [rad]"),
    "input_amplitude_a": ("a_amp", "Accel excitation [m/s²]"),
    "prbs_min_period": ("PRBS_min", "PRBS min hold [steps]"),
    "Q_diag": ("Q", "Output weight diag"),
    "R_diag": ("R", "Input weight diag"),
    "lambda_g": ("λ_g", "Penalty on g"),
    "lambda_y": ("λ_y", "Penalty on σ_y"),
    "lambda_out": ("λ_out", "Output slack penalty"),
    "reg_norm_g": ("‖g‖", "g norm type"),
    "reg_norm_sigma_y": ("‖σ_y‖", "σ_y norm type"),
    "delta_max": ("δ_max", "Max steering [rad]"),
    "a_max": ("a_max", "Max accel [m/s²]"),
    "a_min": ("a_min", "Max braking [m/s²]"),
    "d_delta_max": ("Δδ_max", "Max steer rate [rad/step]"),
    "da_max": ("Δa_max", "Max accel rate [m/s²/step]"),
    "y_lb": ("y_lb", "Output lower bounds"),
    "y_ub": ("y_ub", "Output upper bounds"),
    "v_ref": ("v_ref", "Reference velocity [m/s]"),
    "ref_amplitude": ("A_ref", "Lateral amplitude [m]"),
    "ref_frequency": ("f_ref", "Lateral frequency [Hz]"),
    "L_wheelbase": ("L_wb", "Wheelbase [m]"),
}


def _fmt_val(v: object) -> str:
    if isinstance(v, float):
        if abs(v) < 0.001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}"
    if isinstance(v, list):
        parts = []
        for x in v:
            if isinstance(x, float):
                if abs(x) == float("inf"):
                    parts.append("∞" if x > 0 else "-∞")
                else:
                    parts.append(f"{x:g}")
            else:
                parts.append(str(x))
        return "[" + ", ".join(parts) + "]"
    return str(v)


def _metrics_html(metrics: dict) -> str:
    html = ""
    for group, items in _METRIC_GROUPS.items():
        html += f'<div class="card"><h3>{group}</h3><table>\n'
        for key, label, unit in items:
            val = metrics.get(key)
            if val is None:
                continue
            unit_str = f' <span class="unit">{unit}</span>' if unit else ""
            html += f"<tr><td>{label}</td><td>{_fmt_val(val)}{unit_str}</td></tr>\n"
        html += "</table></div>\n"
    return html


def _config_html(config: DeePCConfig) -> str:
    html = ""
    for group, keys in _CONFIG_GROUPS.items():
        html += f'<div class="card"><h3>{group}</h3><table>\n'
        for key in keys:
            val = getattr(config, key, None)
            if val is None:
                continue
            short, desc = _CONFIG_LABELS.get(key, (key, ""))
            html += (
                f'<tr><td><code>{short}</code>'
                f'<span class="desc">{desc}</span></td>'
                f"<td>{_fmt_val(val)}</td></tr>\n"
            )
        html += "</table></div>\n"
    return html


def build_html(
    tag: str,
    config: DeePCConfig,
    metrics: dict,
    training_div: str,
    sim_div: str,
    wall_time: float,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    metrics_html = _metrics_html(metrics)
    config_html = _config_html(config)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeePC — {tag}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: #f7f8fa; --card: #fff; --border: #e5e7eb;
    --text: #1a1a2e; --muted: #6b7280; --accent: #3b82f6;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 32px 24px; }}

  header {{ margin-bottom: 32px; }}
  header h1 {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }}
  header .meta {{
    display: flex; gap: 16px; margin-top: 6px;
    font-size: 0.82rem; color: var(--muted);
  }}
  header .meta span {{ display: inline-flex; align-items: center; gap: 4px; }}

  section {{ margin-bottom: 28px; }}
  section > h2 {{
    font-size: 1.1rem; font-weight: 600; margin-bottom: 12px;
    padding-bottom: 8px; border-bottom: 2px solid var(--border);
  }}

  .plot-card {{
    background: var(--card); border-radius: 10px; padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 24px;
  }}

  .grid {{ display: grid; gap: 16px; }}
  .grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
  .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
  .grid-4 {{ grid-template-columns: repeat(4, 1fr); }}
  @media (max-width: 900px) {{
    .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }}
  }}

  .card {{
    background: var(--card); border-radius: 10px; padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .card h3 {{
    font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--muted); margin-bottom: 10px;
  }}
  .card table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .card td {{ padding: 4px 0; vertical-align: top; }}
  .card td:first-child {{
    color: var(--text); font-weight: 500; padding-right: 12px;
    white-space: nowrap;
  }}
  .card td:last-child {{
    text-align: right; font-variant-numeric: tabular-nums;
    font-family: "SF Mono", "Consolas", monospace; font-size: 0.83rem;
  }}
  .card tr + tr td {{ border-top: 1px solid var(--border); }}
  .unit {{ color: var(--muted); font-size: 0.75rem; }}
  .desc {{
    display: block; font-size: 0.72rem; color: var(--muted);
    font-weight: 400; line-height: 1.3;
  }}
  code {{
    font-family: "SF Mono", "Consolas", monospace; font-size: 0.85rem;
  }}
</style>
</head>
<body>
<div class="container">

  <header>
    <h1>DeePC Experiment Report</h1>
    <div class="meta">
      <span>{tag}</span>
      <span>{timestamp}</span>
      <span>{wall_time:.1f}s wall time</span>
    </div>
  </header>

  <section>
    <h2>Training Data</h2>
    <div class="plot-card">{training_div}</div>
  </section>

  <section>
    <h2>Simulation Results</h2>
    <div class="plot-card">{sim_div}</div>
  </section>

  <section>
    <h2>Metrics</h2>
    <div class="grid grid-3">{metrics_html}</div>
  </section>

  <section>
    <h2>Configuration</h2>
    <div class="grid grid-4">{config_html}</div>
  </section>

</div>
</body>
</html>"""


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a DeePC experiment and generate an HTML report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Scenario
    p.add_argument("--scenario", type=str, default="default", choices=SCENARIOS,
                   help="Reference trajectory scenario (default: default)")

    # Feature flags
    p.add_argument("--noise-adaptive", action="store_true",
                   help="Enable noise-adaptive regularization")
    p.add_argument("--online-hankel", action="store_true",
                   help="Enable online sliding Hankel window")

    # Data collection
    g = p.add_argument_group("data collection")
    g.add_argument("--T-data", type=int, help="Number of training samples (default: 400)")
    g.add_argument("--noise-std", type=float, help="Measurement noise std (default: 0.01)")
    g.add_argument("--input-amp-delta", type=float,
                   help="Steering excitation amplitude [rad] (default: 0.5)")
    g.add_argument("--input-amp-a", type=float,
                   help="Acceleration excitation amplitude [m/s^2] (default: 1.0)")
    g.add_argument("--prbs-min-period", type=int,
                   help="PRBS minimum hold time [steps] (default: 3)")
    g.add_argument("--seed", type=int, default=42,
                   help="RNG seed for data collection (default: 42)")

    # Horizons
    g = p.add_argument_group("horizons")
    g.add_argument("--Tini", type=int, help="Past window length (default: 3)")
    g.add_argument("--N", type=int, help="Prediction horizon (default: 15)")
    g.add_argument("--sim-duration", type=float, help="Simulation duration [s] (default: 60)")

    # Cost weights
    g = p.add_argument_group("cost weights")
    g.add_argument("--Q", type=float, nargs=3, metavar=("Qx", "Qy", "Qv"),
                   help="Output tracking weight diag [Qx Qy Qv] (default: 10 10 1)")
    g.add_argument("--R", type=float, nargs=2, metavar=("Rd", "Ra"),
                   help="Input cost weight diag [Rd Ra] (default: 0.1 0.1)")

    # Regularization
    g = p.add_argument_group("regularization")
    g.add_argument("--lambda-g", type=float, help="Penalty on g (default: 5)")
    g.add_argument("--lambda-y", type=float, help="Penalty on sigma_y (default: 1e4)")
    g.add_argument("--lambda-out", type=float,
                   help="Penalty on output constraint slack (default: 1e3)")
    g.add_argument("--reg-norm-g", choices=["L1", "L2"],
                   help="g regularization norm (default: L2)")
    g.add_argument("--reg-norm-sigma-y", choices=["L1", "L2"],
                   help="sigma_y regularization norm (default: L1)")

    # Input constraints
    g = p.add_argument_group("input constraints")
    g.add_argument("--delta-max", type=float, help="Max steering angle [rad] (default: 0.5)")
    g.add_argument("--a-max", type=float, help="Max acceleration [m/s^2] (default: 3.0)")
    g.add_argument("--a-min", type=float, help="Max braking [m/s^2] (default: -5.0)")
    g.add_argument("--d-delta-max", type=float, help="Max steering rate [rad/step] (default: 0.1)")
    g.add_argument("--da-max", type=float, help="Max accel rate [m/s^2/step] (default: 0.5)")

    # Output constraints
    g = p.add_argument_group("output constraints")
    g.add_argument("--v-min", type=float, help="Min velocity bound [m/s] (default: 0)")
    g.add_argument("--v-max", type=float, help="Max velocity bound [m/s] (default: 15)")

    # Reference
    g = p.add_argument_group("reference trajectory")
    g.add_argument("--v-ref", type=float, help="Reference velocity [m/s] (default: 5.0)")
    g.add_argument("--ref-amplitude", type=float,
                   help="Sinusoidal lateral amplitude [m] (default: 5.0)")
    g.add_argument("--ref-frequency", type=float,
                   help="Sinusoidal frequency [Hz] (default: 0.05)")

    # Plant
    g = p.add_argument_group("plant")
    g.add_argument("--Ts", type=float, help="Sampling time [s] (default: 0.1)")
    g.add_argument("--L-wheelbase", type=float, help="Wheelbase [m] (default: 2.5)")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> DeePCConfig:
    """Build DeePCConfig from CLI args, only overriding what was passed."""
    overrides = {}

    arg_to_field = {
        "T_data": "T_data",
        "noise_std": "noise_std_output",
        "input_amp_delta": "input_amplitude_delta",
        "input_amp_a": "input_amplitude_a",
        "prbs_min_period": "prbs_min_period",
        "Tini": "Tini",
        "N": "N",
        "sim_duration": "sim_duration",
        "lambda_g": "lambda_g",
        "lambda_y": "lambda_y",
        "lambda_out": "lambda_out",
        "reg_norm_g": "reg_norm_g",
        "reg_norm_sigma_y": "reg_norm_sigma_y",
        "delta_max": "delta_max",
        "a_max": "a_max",
        "a_min": "a_min",
        "d_delta_max": "d_delta_max",
        "da_max": "da_max",
        "v_ref": "v_ref",
        "ref_amplitude": "ref_amplitude",
        "ref_frequency": "ref_frequency",
        "Ts": "Ts",
        "L_wheelbase": "L_wheelbase",
    }

    for arg_name, field_name in arg_to_field.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            overrides[field_name] = val

    if args.Q is not None:
        overrides["Q_diag"] = list(args.Q)
    if args.R is not None:
        overrides["R_diag"] = list(args.R)

    if args.v_min is not None or args.v_max is not None:
        defaults = DeePCConfig()
        y_lb = list(defaults.y_lb)
        y_ub = list(defaults.y_ub)
        if args.v_min is not None:
            y_lb[2] = args.v_min
        if args.v_max is not None:
            y_ub[2] = args.v_max
        overrides["y_lb"] = y_lb
        overrides["y_ub"] = y_ub

    return DeePCConfig(**overrides)


def build_tag(args: argparse.Namespace) -> str:
    """Build a descriptive name for this run."""
    parts = [args.scenario]
    if args.noise_adaptive:
        parts.append("na")
    if args.online_hankel:
        parts.append("oh")
    return "_".join(parts)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    config = build_config(args)
    tag = build_tag(args)
    features = FeatureFlags(
        noise_adaptive=args.noise_adaptive,
        online_hankel=args.online_hankel,
    )

    print(f"=== DeePC run: {tag} ===\n")
    t_wall = time.perf_counter()

    # 1. Training data
    print("Collecting training data...")
    u_data, y_data = collect_data(config, seed=args.seed)
    print(f"  u {u_data.shape}, y {y_data.shape}")
    training_div = plot_training_data(u_data, y_data, config)

    # 2. Reference path
    path = get_reference(args.scenario, config)

    # 3. Simulation
    results = run_simulation(config, features, path_override=path)

    # 4. Metrics
    metrics = compute_metrics(results, tag)

    # 5. Plots
    sim_div = plot_simulation_results(results, config, tag)

    wall_time = time.perf_counter() - t_wall

    # 6. HTML report
    html = build_html(tag, config, metrics, training_div, sim_div, wall_time)

    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{tag}.html"
    report_path.write_text(html)

    # Also save metrics JSON alongside
    metrics_path = out_dir / f"{tag}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nReport: {report_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
