"""DeePC simulation entry point.

Usage (from repo root):
    uv run python -m sim                                    # base (sparse QP)
    uv run python -m sim --noise-adaptive                   # + noise-adaptive lambda
    uv run python -m sim --online-hankel                    # + sliding Hankel window
    uv run python -m sim --noise-adaptive --online-hankel   # both
    uv run python -m sim --scenario hard                    # scenario selection
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from control.config import DeePCConfig
from sim.eval.scenarios import get_reference
from sim.eval.simulation import FeatureFlags, run_simulation, save_results
from sim.eval.visualization import plot_all

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

SCENARIO_SIM_STEPS = {
    "default": 150,
    "hard": 400,
    "circle": 300,
    "square": 350,
    "zigzag": 300,
}


def _build_tag(features: FeatureFlags, scenario: str) -> str:
    """Build a results tag from feature flags and scenario."""
    parts: list[str] = []
    if features.noise_adaptive:
        parts.append("na")
    if features.online_hankel:
        parts.append("oh")
    base = "+".join(parts) if parts else "base"
    if scenario != "default":
        return f"{base}_{scenario}"
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="DeePC simulation")
    parser.add_argument(
        "--noise-adaptive", action="store_true",
        help="Enable noise-adaptive regularization (v5 feature)",
    )
    parser.add_argument(
        "--online-hankel", action="store_true",
        help="Enable online sliding Hankel window (v6 feature)",
    )
    parser.add_argument(
        "--scenario", type=str, default="default",
        choices=["default", "hard", "circle", "square", "zigzag"],
        help="Reference trajectory scenario",
    )
    args = parser.parse_args()

    features = FeatureFlags(
        noise_adaptive=args.noise_adaptive,
        online_hankel=args.online_hankel,
    )
    scenario = args.scenario
    tag = _build_tag(features, scenario)

    sim_steps = SCENARIO_SIM_STEPS[scenario]
    config = DeePCConfig(sim_steps=sim_steps)

    flags_str = []
    if features.noise_adaptive:
        flags_str.append("noise-adaptive")
    if features.online_hankel:
        flags_str.append("online-hankel")
    print(f"=== DeePC | scenario={scenario} | features={flags_str or ['base']} ===")

    y_ref = get_reference(scenario, config)
    results = run_simulation(config, features, y_ref_override=y_ref)

    results_dir = RESULTS_DIR / tag
    save_results(results, tag, results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_all(results, config, save_dir=str(results_dir), tag=tag)

    from sim.comparison.metrics import compute_all_metrics, save_metrics
    metrics = compute_all_metrics(results, tag)
    save_metrics(metrics, results_dir / f"{tag}_metrics.json")

    print("Experiment complete.")


if __name__ == "__main__":
    main()
