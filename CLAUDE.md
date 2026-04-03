## Project

DeePC Sandbox — a research platform for developing, testing, and benchmarking Data-Enabled Predictive Control algorithms across different plant models. The controller uses only pre-collected input-output data (no explicit model) to track reference trajectories via online QP optimization over Hankel matrix representations.

Currently implemented: kinematic bicycle model (autonomous vehicle path tracking).

## Structure

```
run.py              Experiment runner — generates HTML report with interactive plots
tune.py             Bayesian optimization of DeePC parameters (Optuna)
control/            DeePC algorithm (config, QP controller, Hankel matrices, regularization)
plants/             Plant models (currently: kinematic bicycle)
sim/                Simulation logic (data generation, reference scenarios, closed-loop runner)
results/            Output reports, metrics, tuning results
```

## Current State

- Error-based output formulation: outputs are tracking errors relative to a reference (shift-invariant, bounded)
- Stabilized data collection: proportional baseline controller + excitation keeps training errors centered near zero
- Multi-episode training at varied operating points for output diversity
- Sparse QP form (u, y as explicit decision variables, block-diagonal Hessian)
- Bayesian parameter tuning via Optuna (Ts, Tini, N, T_data, lambdas, reg norms)
- Interactive HTML reports with Plotly
- Single plant: kinematic bicycle model with error-based outputs [e_lateral, e_heading, e_velocity]

## Principles

- **Propose before implementing**: describe the change, its benefit, and drawbacks — wait for confirmation on anything non-trivial
- **Do not git commit or push unless explicitly asked**
- **Smallest useful change**: explain tradeoffs if complexity or compute time increases
- **No speculative abstractions**: solve the problem at hand, don't design for hypothetical futures
- **Metrics over intuition**: back claims with numbers (RMSE, condition numbers, solve times)
- **Training data quality matters more than algorithm complexity**: a well-conditioned Hankel matrix with good coverage beats any amount of regularization tuning
- **Plant-agnostic design**: DeePC logic in `control/` should not depend on any specific plant

## Key Lessons Learned

- Absolute/monotonic outputs destroy Hankel conditioning — use error-based or shift-invariant outputs
- Open-loop data collection produces drifted, asymmetric output distributions — a stabilizing baseline during collection keeps errors centered
- Tuning on short simulations and evaluating on long ones exposes drift — the controller must be fundamentally stable, not just tuned to a duration
- Sampling time (Ts) dominates tracking accuracy more than any regularization parameter
- One-at-a-time parameter sweeps miss interactions — Bayesian joint search finds better configurations
