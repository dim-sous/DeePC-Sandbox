## Project

DeePC Sandbox — a research platform for developing, testing, and benchmarking Data-Enabled Predictive Control algorithms across different plant models. The controller uses only pre-collected input-output data (no explicit model) to track reference trajectories via online QP optimization over Hankel matrix representations.

## Structure

```
run.py              Experiment runner — generates HTML report with interactive plots
tune.py             Bayesian optimization of DeePC parameters (Optuna)
control/            DeePC algorithm (config, QP controller, Hankel matrices, regularization)
plants/             Plant models and PlantBase ABC
  base.py           Abstract base class + Constraints/DataCollectionConfig dataclasses
  bicycle_model.py  Kinematic bicycle model (nonlinear, 2 inputs, 3 outputs)
  coupled_masses.py Coupled two-mass spring-damper (LTI MIMO, 2 inputs, 4 outputs)
sim/                Simulation logic (data generation, scenario dispatch, closed-loop runner)
results/            Output reports, metrics, tuning results
```

## Plant Architecture

Every plant inherits from `PlantBase` (in `plants/base.py`) and implements:
- `step(u)`, `reset()`, `get_output(state, reference)` — dynamics and output computation
- `m`, `p`, `Ts`, `input_names`, `output_names` — dimensions and metadata
- `get_constraints()` — named constraint dicts (u_lb, u_ub, du_min, du_max, y_lb, y_ub)
- `get_default_config_overrides()` — plant-appropriate defaults for DeePCConfig
- `get_scenarios()` — reference trajectory generators
- `get_data_collection_config()` — excitation signals, stabilizing controller, episode initial conditions
- `make_episode_initial_state()` — per-episode state initialization
- Optional: custom `plot_training_data()`, `plot_simulation_results()`, `compute_custom_metrics()`

`control/`, `sim/`, `run.py`, and `tune.py` are fully plant-agnostic. Adding a new plant only requires implementing `PlantBase` and registering it in `PLANT_REGISTRY` in `run.py`.

## Current State

- Plant-agnostic architecture: ABC-based, any plant works via `--plant <name>`
- Error-based output formulation: outputs are tracking errors relative to a reference
- Stabilized data collection: proportional baseline controller + excitation keeps training errors centered
- Sparse QP form (u, y as explicit decision variables, block-diagonal Hessian)
- Bayesian parameter tuning via Optuna (Ts, Tini, N, T_data, lambdas, reg norms)
- Interactive HTML reports with Plotly (plant custom or generic auto-plots)
- Plants: kinematic bicycle, coupled two-mass spring-damper (LTI MIMO)

## Principles

- **Propose before implementing**: describe the change, its benefit, and drawbacks — wait for confirmation on anything non-trivial
- **Do not git commit or push unless explicitly asked**
- **Smallest useful change**: explain tradeoffs if complexity or compute time increases
- **No speculative abstractions**: solve the problem at hand, don't design for hypothetical futures
- **Metrics over intuition**: back claims with numbers (RMSE, condition numbers, solve times)
- **Training data quality matters more than algorithm complexity**: a well-conditioned Hankel matrix with good coverage beats any amount of regularization tuning
- **Plant-agnostic design**: DeePC logic in `control/` must not depend on any specific plant

## Key Lessons Learned

- Absolute/monotonic outputs destroy Hankel conditioning — use error-based or shift-invariant outputs
- Open-loop data collection produces drifted, asymmetric output distributions — a stabilizing baseline during collection keeps errors centered
- Tuning on short simulations and evaluating on long ones exposes drift — the controller must be fundamentally stable, not just tuned to a duration
- Sampling time (Ts) dominates tracking accuracy more than any regularization parameter
- One-at-a-time parameter sweeps miss interactions — Bayesian joint search finds better configurations
