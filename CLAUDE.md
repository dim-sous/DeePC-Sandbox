## Current State
- Completed: v1
- Active: v1 (gated)
- Next: v2
- Frozen (do not modify): None

## If context is unclear Re-read this file top to bottom. Ask me to confirm the current version.

DeePC sandbox — general-purpose platform for developing, testing, and benchmarking Data-Enabled Predictive Control algorithms.
Plants live in `plants/`, shared across versions. Version folders contain only DeePC logic.

## Principles
- **Incremental versioning**: one or few changes per version, baseline always reproducible
- **Three-stage gate**: (A) validation, (B) evaluation, (C) stress testing — identical scenarios across versions
- **Propose before implementing**: describe problem, benefit, drawbacks; wait for confirmation on non-trivial changes
- **Production-grade**: robustness > numerical stability > efficiency > maintainability > reproducibility
- **Smallest useful change**: explain tradeoffs if complexity or compute time increases significantly
- **Interpretability**: transparent optimization structure, diagnostics that explain behavior
- **Modular**: plants · data generation · data matrices · optimization · evaluation · stress testing
- **Benchmarking**: every version compared against v1 and prior version

## Candidate Upgrades
- Regularized / robust DeePC formulations
- Output scaling and QP conditioning
- Noise handling improvements
- Constraint handling (output constraints, input rate constraints)
- Dataset management and online updating
- Computational improvements for real-time use
- PID / MPC baselines
- Additional plant models
