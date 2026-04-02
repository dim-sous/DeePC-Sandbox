## Current State
- Completed: v1
- Active: v1 (3 stage gating needed)
- Next: v2
- Frozen (do not modify): None

## If context is unclear Re-read this file top to bottom. Ask me to confirm the current version.

You are assisting in the development of a DeePC sandbox — a general-purpose platform for developing, testing, and benchmarking Data-Enabled Predictive Control algorithms.
Version 1 (proof-of-concept) is implemented and working in closed-loop simulation using a kinematic bicycle model as the first example plant. All further development should follow an incremental engineering process toward a robust, reusable DeePC library — not a research prototype.
Plant models live in the top-level `plants/` directory and are shared across versions. Version folders contain only DeePC logic and import plants from `plants/`.
────────────────────────────────────────
CORE PRINCIPLES
────────────────────────────────────────
INCREMENTAL VERSIONING
Each improvement creates a new version (v1 → v2 → v3 → ...). Introduce one or a small number of changes per version. The baseline must always remain reproducible.
THREE-STAGE GATE — run before proceeding to the next version
A. Validation — confirm correctness and mathematical consistency
B. Evaluation — measure tracking error, control effort, solver time, constraint violations, stability
C. Stress Testing — test under noise, disturbances, poor excitation, actuator limits, aggressive references, nonlinear regimes, changing conditions
Use identical scenarios across versions so comparisons are fair.
PROPOSE BEFORE IMPLEMENTING
Before writing any new code, output a short proposal: what problem does this address, why is it useful, expected benefits and drawbacks. For non-trivial changes, wait for confirmation before proceeding. If you identify a meaningful tradeoff or alternative approach, surface it first.
PRODUCTION-GRADE DECISIONS
Prioritize: robustness · numerical stability · computational efficiency · maintainability · reproducibility. Avoid purely academic features that add complexity without clear practical benefit.
AVOID OVER-ENGINEERING
Prefer the smallest change that meaningfully improves performance. If a feature significantly increases complexity or computation time, explain the tradeoff and suggest lighter alternatives.
MAINTAIN INTERPRETABILITY
The controller should remain transparent. Prefer formulations where system behavior, optimization structure, and data usage are easy to analyze. Diagnostics and visualization tools should explain why the controller behaves as it does.
COMPUTATIONAL FEASIBILITY
For every new feature, estimate the effect on solver time and scaling behavior. Ensure the controller remains feasible for real-time operation.
MODULAR STRUCTURE
Keep these components cleanly separated: plant models (in `plants/`) · data generation · Hankel matrix construction · DeePC optimization · evaluation and benchmarking · stress testing scenarios. Version folders should only contain DeePC-specific code and import plant models from `plants/`.
BENCHMARKING
Every version must be compared against the v1 baseline and the immediately prior version across: tracking accuracy · control effort · solver time · robustness under stress scenarios. Evaluation scripts must be reproducible. Stress-test scenarios should be defined as reusable fixtures early in the project.
────────────────────────────────────────
CANDIDATE UPGRADE DIRECTIONS
────────────────────────────────────────
Regularized DeePC formulations
Improved handling of noisy measurements
Better Hankel matrix conditioning
Constraint handling improvements
Dataset management and online updating
Computational improvements for real-time use
Robustness to insufficient or poorly excited datasets
Comparison against PID or MPC baselines
Additional plant models (marine vessels, quadrotors, process control systems)
────────────────────────────────────────
The goal is a well-tested, maintainable DeePC platform that improves incrementally — not one that is maximally sophisticated.
────────────────────────────────────────
