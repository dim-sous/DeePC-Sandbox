# Backlog

## Achievements

- **Error-based output formulation** — switching from absolute/monotonic outputs to tracking errors reduced Hankel conditioning from 55,000 to 205 and made the controller shift-invariant
- **Stabilized data collection** — adding a proportional baseline controller during training keeps errors bounded near zero (std 0.26m vs 15.6m without), producing symmetric, well-distributed training data
- **Multi-episode training** — collecting data across multiple episodes at varied operating points gives the output Hankel matrix coverage over the operating regime
- **Sparse QP form** — explicit u, y decision variables with equality constraints to Hankel, giving a block-diagonal Hessian that is well-conditioned regardless of output scale
- **Bayesian parameter tuning** — Optuna TPE search over the joint parameter space found configurations that one-at-a-time sweeps missed
- **Sub-meter lateral tracking** — 0.54m lateral RMSE over 60 seconds on the bicycle model

## Known Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| Position drift | High | Per-step errors are small but accumulate over time due to no integral action. Fundamental limitation of vanilla DeePC |
| LTI assumption | Medium | Willems' lemma assumes LTI; nonlinear plants (e.g. bicycle at high speed/steering) degrade outside the linearization regime covered by training data |
| Solver warnings | Low | OSQP occasionally reports "solution may be inaccurate" for certain parameter combinations |
| Solve time variance | Medium | Solve time varies 5-400ms. Not real-time guaranteed without a fixed-iteration solver |
| No disturbance rejection | Medium | Persistent disturbances are absorbed as slack rather than compensated |

## What Worked

- Error-based outputs over absolute outputs — single biggest improvement
- Stabilized data collection — essential for training data quality
- Smaller Ts dominates all other parameters for tracking accuracy
- L2 norm on g, L1 norm on sigma_y — smooth control with exact penalty on past-data mismatch
- High lambda_y forces near-exact past-data consistency, which is correct when training data quality is good
- Bayesian joint parameter search over one-at-a-time sweeps

## What Didn't Work

- Absolute position outputs — non-stationary, destroys Hankel conditioning
- Incremental outputs (dx, dy) — bounded but still accumulates drift without position feedback
- Open-loop data collection — plant wanders during excitation, producing asymmetric error distributions
- One-at-a-time parameter sweeps — misses interactions, finds suboptimal configs
- Tuning on short simulations — parameters that minimize 10s RMSE can diverge at 60s
- Large prediction horizons — N > 10 doesn't help and slows down the solver

## Ideas for Future Work

### High Priority

- **Integral action** — augment outputs with accumulated error to prevent drift
- **Measurement noise sensitivity** — current noise_std=0.01 is optimistic; test with realistic sensor noise levels
- **MPC baseline** — model-based MPC to establish the performance ceiling and quantify the cost of being model-free
- **PID baseline** — simplest controller for comparison
- **Additional plant models** — marine vessel, quadrotor, thermal system, process control to test DeePC generality

### Medium Priority

- **Disturbance feedforward** — estimate persistent disturbances from prediction residuals
- **Online Hankel adaptation** — sliding window exists but degrades nominal tracking; needs better data quality filtering
- **Robust constraint tightening** — output constraints that account for noise (Wasserstein ball formulation)

### Low Priority

- **Low-rank SVD approximation** of Hankel for computational efficiency
- **Dataset management** — save/load pre-collected datasets with metadata
- **Real-time solver** — fixed-iteration ADMM or explicit MPC for guaranteed solve times
- **CI pipeline** — tests, type checking, linting

## Pros and Cons of DeePC

### Pros

- No model identification — apply excitation, record data, control
- Handles unknown dynamics within the training regime
- Clean mathematical formulation (QP at each step)
- Regularization provides natural robustness to noise
- Works well for narrow operating regimes with good data

### Cons

- Training data quality is the bottleneck — requires careful excitation design, stabilized collection, and regime coverage
- LTI assumption limits applicability to nonlinear systems
- No integral action — error drift is inherent
- Computational cost scales with data volume (Hankel matrix size)
- Fragile to distribution shift — performance degrades sharply outside training regime
- The "model-free" promise is misleading — model identification effort is traded for equally difficult data engineering effort
