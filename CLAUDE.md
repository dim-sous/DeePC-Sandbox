## Current State
- Completed: v1, v2, v3, v4, v5, v6
- Active: v6 (gated — A 17/17, B 15/15, C 4/9)
- Next: v7
- Frozen (do not modify): v1_baseline, v2, v3, v4, v5, v6

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
- **Do not git commit or push unless explicitly asked**

## Candidate Upgrades
- Startup initialization fix (K7 — buffer/sim consistency at arbitrary initial conditions)
- ~~Online sliding Hankel window for regime adaptation~~ — implemented in v6 (append-then-slide)
- ~~Noise handling / robust DeePC formulations~~ — implemented in v5 (noise-adaptive regularization)
- Low-rank SVD approximation of Hankel matrix
- Dataset management and online updating
- Data denoising before Hankel construction
- PID / MPC baselines
- Additional plant models
