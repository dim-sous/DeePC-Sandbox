## Current State
- Single unified platform: `control/` (DeePC logic) + `sim/` (simulation, evaluation, comparison)
- Base: sparse QP form (v4) with optional features via CLI flags
- Features: `--noise-adaptive` (v5), `--online-hankel` (v6)
- Gate: `uv run python -m sim.gate` (A/B/C across flag combos)

## If context is unclear Re-read this file top to bottom. Ask me to confirm the current state.

DeePC sandbox — general-purpose platform for developing, testing, and benchmarking Data-Enabled Predictive Control algorithms.
Plants live in `plants/`, shared. Control logic in `control/`. Simulation and evaluation in `sim/`.

## Principles
- **Three-stage gate**: (A) validation, (B) evaluation, (C) stress testing — identical scenarios across flag combos
- **Propose before implementing**: describe problem, benefit, drawbacks; wait for confirmation on non-trivial changes
- **Production-grade**: robustness > numerical stability > efficiency > maintainability > reproducibility
- **Smallest useful change**: explain tradeoffs if complexity or compute time increases significantly
- **Interpretability**: transparent optimization structure, diagnostics that explain behavior
- **Modular**: plants · data generation · data matrices · optimization · evaluation · stress testing
- **Do not git commit or push unless explicitly asked**

## Candidate Upgrades
- Startup initialization fix (K7 — buffer/sim consistency at arbitrary initial conditions)
- Low-rank SVD approximation of Hankel matrix
- Dataset management and online updating
- Data denoising before Hankel construction
- PID / MPC baselines
- Additional plant models
