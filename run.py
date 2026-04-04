"""Standalone DeePC experiment runner.

Generates training data, builds a reference trajectory, runs closed-loop
simulation, computes metrics, and produces a self-contained HTML report
with interactive Plotly charts.

Usage (from repo root):
    uv run python run.py                          # defaults (bicycle)
    uv run python run.py --plant bicycle --scenario lissajous
    uv run python run.py --N 20 --lambda-g 10
    uv run python run.py --Q 10 10 1 --R 0.1 0.1
    uv run python run.py --help                   # full list
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from control.config import DeePCConfig, build_deepc_config
from plants.base import PlantBase
from plants.bicycle_model import BicycleModel
from plants.coupled_masses import CoupledMasses
from sim.data_generation import collect_data
from sim.scenarios import get_reference
from sim.simulation import FeatureFlags, run_simulation

REPO_ROOT = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"

# ── Plant registry ───────────────────────────────────────────────────

PLANT_REGISTRY: dict[str, type[PlantBase]] = {
    "bicycle": BicycleModel,
    "coupled_masses": CoupledMasses,
}

# ── Plot style ───────────────────────────────────────────────────────

_C_REF = "#d62728"
_C_ACT = "#1f77b4"
_C_BAND = "#aaaaaa"
_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

_PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=50, r=30, t=40, b=40),
    hoverlabel=dict(font_size=11),
)


# ── Generic metrics ──────────────────────────────────────────────────

def compute_metrics(
    results: dict, plant: PlantBase, tag: str,
) -> dict[str, float | str]:
    """Compute performance metrics from simulation results."""
    metrics: dict[str, float | str] = {"version": tag}

    # Per-channel output RMSE
    errors = np.asarray(results["y_history"])
    for i, name in enumerate(plant.output_names):
        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean(errors[:, i] ** 2)))

    # Plant-specific custom metrics (e.g. position RMSE for bicycle)
    metrics.update(plant.compute_custom_metrics(results))

    # Solver stats
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


# ── Generic Plotly figures ───────────────────────────────────────────

def generic_plot_training_data(
    u_data: np.ndarray,
    y_data: np.ndarray,
    plant: PlantBase,
    config: DeePCConfig,
) -> str:
    """Generic training data plot: one subplot per input/output channel."""
    T = len(u_data)
    t = (np.arange(T) * config.Ts).tolist()
    m, p = plant.m, plant.p
    total = m + p

    cols = min(total, 3)
    rows = (total + cols - 1) // cols
    titles = [f"Input: {n}" for n in plant.input_names] + \
             [f"Output: {n}" for n in plant.output_names]

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.08, vertical_spacing=0.12)

    for idx in range(total):
        r = idx // cols + 1
        c = idx % cols + 1
        color = _COLORS[idx % len(_COLORS)]
        if idx < m:
            fig.add_trace(go.Scatter(
                x=t, y=u_data[:, idx].tolist(), mode="lines",
                line=dict(color=color, width=1),
                name=plant.input_names[idx], showlegend=False,
            ), row=r, col=c)
        else:
            oi = idx - m
            fig.add_trace(go.Scatter(
                x=t, y=y_data[:, oi].tolist(), mode="lines",
                line=dict(color=color, width=1),
                name=plant.output_names[oi], showlegend=False,
            ), row=r, col=c)
        fig.update_xaxes(title_text="Time [s]", row=r, col=c)

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=280 * rows,
        showlegend=False,
        title=dict(text=f"Training Data (T={T})", x=0.5, font_size=15),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def generic_plot_simulation_results(
    results: dict,
    plant: PlantBase,
    config: DeePCConfig,
) -> str:
    """Generic simulation results plot: outputs + inputs time series."""
    errors = np.asarray(results["y_history"])
    u_hist = np.asarray(results["u_history"])
    times = np.asarray(results["times"])
    m, p = plant.m, plant.p

    n = min(len(times), len(errors), len(u_hist))
    times_list = times[:n].tolist()
    errors, u_hist = errors[:n], u_hist[:n]

    total = p + m
    cols = min(total, 3)
    rows = (total + cols - 1) // cols
    titles = [f"Output: {n}" for n in plant.output_names] + \
             [f"Input: {n}" for n in plant.input_names]

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.08, vertical_spacing=0.10)

    for idx in range(total):
        r = idx // cols + 1
        c = idx % cols + 1
        color = _COLORS[idx % len(_COLORS)]
        if idx < p:
            # Output error (reference is zero)
            fig.add_trace(go.Scatter(
                x=times_list, y=errors[:, idx].tolist(), mode="lines",
                line=dict(color=color, width=1.2),
                name=plant.output_names[idx], showlegend=False,
            ), row=r, col=c)
            fig.add_hline(y=0, line=dict(color=_C_BAND, dash="dash", width=1),
                          row=r, col=c)
        else:
            ii = idx - p
            fig.add_trace(go.Scatter(
                x=times_list, y=u_hist[:, ii].tolist(), mode="lines",
                line=dict(color=color, width=1.2),
                name=plant.input_names[ii], showlegend=False,
            ), row=r, col=c)
        fig.update_xaxes(title_text="Time [s]", row=r, col=c)

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=280 * rows,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── HTML report ──────────────────────────────────────────────────────

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
                    parts.append("\u221e" if x > 0 else "-\u221e")
                else:
                    parts.append(f"{x:g}")
            else:
                parts.append(str(x))
        return "[" + ", ".join(parts) + "]"
    return str(v)


def _metrics_html(metrics: dict, plant: PlantBase) -> str:
    html = ""

    # Per-channel tracking errors
    html += '<div class="card"><h3>Tracking (errors)</h3><table>\n'
    for name in plant.output_names:
        key = f"rmse_{name}"
        val = metrics.get(key)
        if val is not None:
            html += f"<tr><td>{name} RMSE</td><td>{_fmt_val(val)}</td></tr>\n"
    html += "</table></div>\n"

    # Custom metrics from plant
    custom = {k: v for k, v in metrics.items()
              if not k.startswith("rmse_") and k not in (
                  "version", "avg_solve_time_s", "max_solve_time_s",
                  "total_solve_time_s", "optimal_solve_pct",
                  "mean_sigma_y_norm", "max_sigma_y_norm",
              )}
    if custom:
        html += '<div class="card"><h3>Custom Metrics</h3><table>\n'
        for key, val in custom.items():
            html += f"<tr><td>{key}</td><td>{_fmt_val(val)}</td></tr>\n"
        html += "</table></div>\n"

    # Solver
    html += '<div class="card"><h3>Solver</h3><table>\n'
    for key, label, unit in [
        ("optimal_solve_pct", "Optimal Solves", "%"),
        ("avg_solve_time_s", "Avg Solve Time", "s"),
        ("max_solve_time_s", "Max Solve Time", "s"),
        ("total_solve_time_s", "Total Solve Time", "s"),
    ]:
        val = metrics.get(key)
        if val is not None:
            unit_str = f' <span class="unit">{unit}</span>' if unit else ""
            html += f"<tr><td>{label}</td><td>{_fmt_val(val)}{unit_str}</td></tr>\n"
    html += "</table></div>\n"

    # Constraints
    html += '<div class="card"><h3>Constraints</h3><table>\n'
    for key, label in [
        ("mean_sigma_y_norm", "Mean Slack \u2016\u03c3_y\u2016"),
        ("max_sigma_y_norm", "Max Slack \u2016\u03c3_y\u2016"),
    ]:
        val = metrics.get(key)
        if val is not None:
            html += f"<tr><td>{label}</td><td>{_fmt_val(val)}</td></tr>\n"
    html += "</table></div>\n"

    return html


_CONFIG_DISPLAY = {
    "Tini": ("T_ini", "Past window"),
    "N": ("N", "Prediction horizon"),
    "sim_duration": ("T_sim", "Simulation duration [s]"),
    "Ts": ("T_s", "Sampling time [s]"),
    "T_data": ("T_data", "Training samples"),
    "noise_std_output": ("\u03c3_noise", "Measurement noise std"),
    "Q_diag": ("Q", "Output weight diag"),
    "R_diag": ("R", "Input weight diag"),
    "lambda_g": ("\u03bb_g", "Penalty on g"),
    "lambda_y": ("\u03bb_y", "Penalty on \u03c3_y"),
    "lambda_out": ("\u03bb_out", "Output slack penalty"),
    "reg_norm_g": ("\u2016g\u2016", "g norm type"),
    "reg_norm_sigma_y": ("\u2016\u03c3_y\u2016", "\u03c3_y norm type"),
}

_CONFIG_GROUPS = {
    "Horizons & Simulation": ["Tini", "N", "sim_duration", "Ts"],
    "Data Collection": ["T_data", "noise_std_output"],
    "Cost Weights": ["Q_diag", "R_diag"],
    "Regularization": ["lambda_g", "lambda_y", "lambda_out",
                        "reg_norm_g", "reg_norm_sigma_y"],
}


def _config_html(config: DeePCConfig) -> str:
    html = ""
    for group, keys in _CONFIG_GROUPS.items():
        html += f'<div class="card"><h3>{group}</h3><table>\n'
        for key in keys:
            val = getattr(config, key, None)
            if val is None:
                continue
            short, desc = _CONFIG_DISPLAY.get(key, (key, ""))
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
    plant: PlantBase,
    training_div: str,
    sim_div: str,
    wall_time: float,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    metrics_html = _metrics_html(metrics, plant)
    config_html = _config_html(config)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeePC \u2014 {tag}</title>
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


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a DeePC experiment and generate an HTML report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Plant selection
    p.add_argument("--plant", type=str, default="bicycle",
                   choices=list(PLANT_REGISTRY.keys()),
                   help="Plant model to use (default: bicycle)")
    p.add_argument("--scenario", type=str, default="default",
                   help="Reference trajectory scenario (default: default)")

    # Feature flags
    p.add_argument("--noise-adaptive", action="store_true",
                   help="Enable noise-adaptive regularization")
    p.add_argument("--online-hankel", action="store_true",
                   help="Enable online sliding Hankel window")
    p.add_argument("--no-constraints", action="store_true",
                   help="Disable all hard constraints (input, rate, output)")

    # Data collection
    g = p.add_argument_group("data collection")
    g.add_argument("--T-data", type=int, help="Number of training samples")
    g.add_argument("--noise-std", type=float, dest="noise_std_output",
                   help="Measurement noise std")
    g.add_argument("--seed", type=int, default=42,
                   help="RNG seed for data collection (default: 42)")

    # Horizons
    g = p.add_argument_group("horizons")
    g.add_argument("--Tini", type=int, help="Past window length")
    g.add_argument("--N", type=int, help="Prediction horizon")
    g.add_argument("--sim-duration", type=float, help="Simulation duration [s]")
    g.add_argument("--Ts", type=float, help="Sampling time [s]")

    # Cost weights (variable length for any plant)
    g = p.add_argument_group("cost weights")
    g.add_argument("--Q", type=float, nargs="+", dest="Q_diag",
                   help="Output tracking weight diag (one value per output)")
    g.add_argument("--R", type=float, nargs="+", dest="R_diag",
                   help="Input cost weight diag (one value per input)")

    # Regularization
    g = p.add_argument_group("regularization")
    g.add_argument("--lambda-g", type=float, help="Penalty on g")
    g.add_argument("--lambda-y", type=float, help="Penalty on sigma_y")
    g.add_argument("--lambda-out", type=float,
                   help="Penalty on output constraint slack")
    g.add_argument("--reg-norm-g", choices=["L1", "L2"],
                   help="g regularization norm")
    g.add_argument("--reg-norm-sigma-y", choices=["L1", "L2"],
                   help="sigma_y regularization norm")

    return p.parse_args()


def build_config(args: argparse.Namespace, plant: PlantBase) -> DeePCConfig:
    """Build DeePCConfig from plant defaults + CLI overrides."""
    overrides: dict = {}

    field_map = {
        "T_data": "T_data",
        "noise_std_output": "noise_std_output",
        "Tini": "Tini",
        "N": "N",
        "sim_duration": "sim_duration",
        "Ts": "Ts",
        "lambda_g": "lambda_g",
        "lambda_y": "lambda_y",
        "lambda_out": "lambda_out",
        "reg_norm_g": "reg_norm_g",
        "reg_norm_sigma_y": "reg_norm_sigma_y",
    }

    for arg_name, field_name in field_map.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            overrides[field_name] = val

    if args.Q_diag is not None:
        overrides["Q_diag"] = list(args.Q_diag)
    if args.R_diag is not None:
        overrides["R_diag"] = list(args.R_diag)

    return build_deepc_config(plant, **overrides)


def build_tag(args: argparse.Namespace) -> str:
    """Build a descriptive name for this run."""
    parts = [args.plant, args.scenario]
    if args.noise_adaptive:
        parts.append("na")
    if args.online_hankel:
        parts.append("oh")
    if args.no_constraints:
        parts.append("nocon")
    return "_".join(parts)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Instantiate the plant
    plant_cls = PLANT_REGISTRY[args.plant]
    plant = plant_cls()

    # Validate scenario
    scenarios = plant.get_scenarios()
    if args.scenario not in scenarios:
        available = ", ".join(sorted(scenarios))
        raise SystemExit(
            f"Unknown scenario '{args.scenario}'. "
            f"Available for {args.plant}: {available}"
        )

    config = build_config(args, plant)
    tag = build_tag(args)
    features = FeatureFlags(
        noise_adaptive=args.noise_adaptive,
        online_hankel=args.online_hankel,
        no_constraints=args.no_constraints,
    )

    print(f"=== DeePC run: {tag} ===\n")
    t_wall = time.perf_counter()

    # 1. Training data
    print("Collecting training data...")
    u_data, y_data = collect_data(plant, config, seed=args.seed)
    print(f"  u {u_data.shape}, y {y_data.shape}")

    # Training data plot (plant custom or generic)
    training_div = plant.plot_training_data(u_data, y_data, config.Ts)
    if training_div is None:
        training_div = generic_plot_training_data(u_data, y_data, plant, config)

    # 2. Reference path
    path = get_reference(plant, args.scenario, config)

    # 3. Simulation
    results = run_simulation(plant, config, features, u_data, y_data, path)

    # 4. Metrics
    metrics = compute_metrics(results, plant, tag)

    # 5. Simulation plot (plant custom or generic)
    sim_div = plant.plot_simulation_results(results, config)
    if sim_div is None:
        sim_div = generic_plot_simulation_results(results, plant, config)

    wall_time = time.perf_counter() - t_wall

    # 6. HTML report
    html = build_html(tag, config, metrics, plant, training_div, sim_div, wall_time)

    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{tag}.html"
    report_path.write_text(html)

    metrics_path = out_dir / f"{tag}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nReport: {report_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
