"""Bayesian optimization of DeePC tuning parameters using Optuna.

Searches the joint parameter space to minimize position tracking RMSE.

Usage:
    uv run python tune.py
    uv run python tune.py --n-trials 200
    uv run python tune.py --n-trials 50 --sim-duration 5
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

import numpy as np
import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from control.config import DeePCConfig
from sim.scenarios import get_reference
from sim.simulation import FeatureFlags, run_simulation
from run import compute_metrics

REPO_ROOT = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results" / "tune"


def run_trial(config: DeePCConfig) -> dict:
    """Run one simulation silently and return metrics."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        path = get_reference("default", config)
        results = run_simulation(config, FeatureFlags(), path_override=path)
        return compute_metrics(results, "trial")
    finally:
        sys.stdout = old_stdout


def objective(trial: optuna.Trial, sim_duration: float) -> float:
    """Optuna objective: minimize position RMSE."""
    params = {
        "sim_duration": sim_duration,
        "Ts": trial.suggest_float("Ts", 0.02, 0.2, log=True),
        "Tini": trial.suggest_int("Tini", 1, 10),
        "N": trial.suggest_int("N", 3, 30),
        "T_data": trial.suggest_int("T_data", 100, 1000, step=50),
        "lambda_g": trial.suggest_float("lambda_g", 0.01, 200.0, log=True),
        "lambda_y": trial.suggest_float("lambda_y", 1.0, 1e6, log=True),
        "reg_norm_g": trial.suggest_categorical("reg_norm_g", ["L1", "L2"]),
        "reg_norm_sigma_y": trial.suggest_categorical("reg_norm_sigma_y", ["L1", "L2"]),
    }

    try:
        config = DeePCConfig(**params)
        metrics = run_trial(config)
        rmse = metrics["rmse_position"]
        if np.isnan(rmse) or np.isinf(rmse):
            return 1e6

        # Log individual metrics for analysis
        trial.set_user_attr("rmse_lateral", metrics["rmse_lateral"])
        trial.set_user_attr("rmse_heading", metrics["rmse_heading"])
        trial.set_user_attr("rmse_v", metrics["rmse_v"])
        trial.set_user_attr("rmse_position", rmse)
        trial.set_user_attr("optimal_solve_pct", metrics.get("optimal_solve_pct", 0))

        return rmse

    except Exception as e:
        trial.set_user_attr("error", str(e))
        return 1e6


# ── HTML report ───────────────────────────────────────────────────────

PARAM_NAMES = ["Ts", "Tini", "N", "T_data", "lambda_g", "lambda_y",
               "reg_norm_g", "reg_norm_sigma_y"]


def build_report(study: optuna.Study, wall_time: float) -> str:
    """Build HTML report with optimization results."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Best trial summary
    best = study.best_trial
    best_params_rows = ""
    for k, v in best.params.items():
        if isinstance(v, float):
            best_params_rows += f"<tr><td>{k}</td><td>{v:.6g}</td></tr>\n"
        else:
            best_params_rows += f"<tr><td>{k}</td><td>{v}</td></tr>\n"
    best_metrics_rows = ""
    for k in ["rmse_position", "rmse_lateral", "rmse_heading", "rmse_v", "optimal_solve_pct"]:
        v = best.user_attrs.get(k)
        if v is not None:
            best_metrics_rows += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>\n"

    # Optimization history plot
    trials = [t for t in study.trials if t.value is not None and t.value < 1e5]
    trial_nums = [t.number for t in trials]
    trial_vals = [t.value for t in trials]
    best_so_far = []
    bsf = float("inf")
    for v in trial_vals:
        bsf = min(bsf, v)
        best_so_far.append(bsf)

    fig_history = go.Figure()
    fig_history.add_trace(go.Scatter(
        x=trial_nums, y=trial_vals, mode="markers",
        marker=dict(size=4, color="#aaa"), name="Trial",
    ))
    fig_history.add_trace(go.Scatter(
        x=trial_nums, y=best_so_far, mode="lines",
        line=dict(color="#d62728", width=2), name="Best so far",
    ))
    fig_history.update_layout(
        template="plotly_white", height=350,
        title="Optimization History",
        xaxis_title="Trial", yaxis_title="Position RMSE [m]",
        margin=dict(l=50, r=30, t=40, b=40),
    )
    history_div = fig_history.to_html(full_html=False, include_plotlyjs=False)

    # Parameter importance via scatter plots
    numeric_params = [p for p in PARAM_NAMES if p not in ("reg_norm_g", "reg_norm_sigma_y")]
    n_params = len(numeric_params)
    fig_params = make_subplots(
        rows=2, cols=3,
        subplot_titles=numeric_params,
        horizontal_spacing=0.08, vertical_spacing=0.12,
    )
    positions = [(r, c) for r in range(1, 3) for c in range(1, 4)]
    for i, pname in enumerate(numeric_params):
        row, col = positions[i]
        x_vals = [t.params.get(pname) for t in trials]
        fig_params.add_trace(go.Scatter(
            x=x_vals, y=trial_vals, mode="markers",
            marker=dict(size=4, color=trial_vals, colorscale="Viridis",
                        showscale=(i == 0)),
            showlegend=False,
        ), row=row, col=col)
        fig_params.update_xaxes(
            title_text=pname, row=row, col=col,
            type="log" if pname in ("Ts", "lambda_g", "lambda_y") else "linear",
        )
        fig_params.update_yaxes(title_text="RMSE [m]", row=row, col=col)

    fig_params.update_layout(
        template="plotly_white", height=500,
        title="Parameter vs Objective",
        margin=dict(l=50, r=30, t=40, b=40),
    )
    params_div = fig_params.to_html(full_html=False, include_plotlyjs=False)

    # Categorical params: box plots
    fig_cat = make_subplots(rows=1, cols=2,
                            subplot_titles=["reg_norm_g", "reg_norm_sigma_y"])
    for i, pname in enumerate(["reg_norm_g", "reg_norm_sigma_y"]):
        for cat in ["L1", "L2"]:
            vals = [t.value for t in trials if t.params.get(pname) == cat]
            fig_cat.add_trace(go.Box(
                y=vals, name=cat, showlegend=False,
            ), row=1, col=i + 1)
        fig_cat.update_yaxes(title_text="RMSE [m]", row=1, col=i + 1)

    fig_cat.update_layout(
        template="plotly_white", height=300,
        title="Regularization Norm Effect",
        margin=dict(l=50, r=30, t=40, b=40),
    )
    cat_div = fig_cat.to_html(full_html=False, include_plotlyjs=False)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeePC Tuning Results</title>
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
  header h1 {{ font-size: 1.5rem; font-weight: 700; }}
  header .meta {{
    display: flex; gap: 16px; margin-top: 6px;
    font-size: 0.82rem; color: var(--muted);
  }}
  section {{ margin-bottom: 24px; }}
  section > h2 {{
    font-size: 1.1rem; font-weight: 600; margin-bottom: 12px;
    padding-bottom: 8px; border-bottom: 2px solid var(--border);
  }}
  .plot-card {{
    background: var(--card); border-radius: 10px; padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 16px;
  }}
  .grid {{ display: grid; gap: 16px; grid-template-columns: 1fr 1fr; }}
  @media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{
    background: var(--card); border-radius: 10px; padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .card h3 {{
    font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--muted); margin-bottom: 10px;
  }}
  .card table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .card td {{ padding: 4px 0; }}
  .card td:first-child {{ font-weight: 500; padding-right: 12px; }}
  .card td:last-child {{
    text-align: right; font-family: "SF Mono", "Consolas", monospace;
    font-size: 0.83rem;
  }}
  .card tr + tr td {{ border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>DeePC Bayesian Optimization Results</h1>
    <div class="meta">
      <span>{timestamp}</span>
      <span>{len(study.trials)} trials</span>
      <span>{wall_time:.0f}s total</span>
      <span>Best RMSE: {best.value:.4f} m</span>
    </div>
  </header>

  <section>
    <h2>Best Configuration</h2>
    <div class="grid">
      <div class="card">
        <h3>Parameters</h3>
        <table>{best_params_rows}</table>
      </div>
      <div class="card">
        <h3>Metrics</h3>
        <table>{best_metrics_rows}</table>
      </div>
    </div>
  </section>

  <section>
    <h2>Optimization History</h2>
    <div class="plot-card">{history_div}</div>
  </section>

  <section>
    <h2>Parameter Sensitivity</h2>
    <div class="plot-card">{params_div}</div>
    <div class="plot-card">{cat_div}</div>
  </section>

</div>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bayesian optimization for DeePC tuning")
    p.add_argument("--n-trials", type=int, default=100,
                   help="Number of optimization trials (default: 100)")
    p.add_argument("--sim-duration", type=float, default=10.0,
                   help="Simulation duration per trial [s] (default: 10)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"=== DeePC Bayesian Optimization ===")
    print(f"  trials={args.n_trials}, sim_duration={args.sim_duration}s")
    print()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    t_wall = time.perf_counter()

    def obj(trial):
        return objective(trial, args.sim_duration)

    study.optimize(obj, n_trials=args.n_trials, show_progress_bar=True)

    wall_time = time.perf_counter() - t_wall

    # Print summary
    best = study.best_trial
    print(f"\n{'=' * 50}")
    print(f"  Best trial #{best.number}: RMSE = {best.value:.4f} m")
    print(f"{'=' * 50}")
    for k, v in best.params.items():
        print(f"  {k:25s}  {v}")
    print()

    # Save report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    html = build_report(study, wall_time)
    report_path = RESULTS_DIR / "tune.html"
    report_path.write_text(html)

    # Save best params as JSON
    best_data = {
        "best_value": best.value,
        "best_params": best.params,
        "best_metrics": dict(best.user_attrs),
        "n_trials": len(study.trials),
        "wall_time_s": round(wall_time, 1),
    }
    json_path = RESULTS_DIR / "tune.json"
    with open(json_path, "w") as f:
        json.dump(best_data, f, indent=2)

    print(f"Report: {report_path}")
    print(f"Best params: {json_path}")
    print(f"Wall time: {wall_time:.0f}s")


if __name__ == "__main__":
    main()
