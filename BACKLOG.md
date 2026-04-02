# DeePC Sandbox -- Backlog

Items are grouped by theme and ordered by priority within each group.
Regularization (`lambda_g`, `lambda_y`) is treated as a robustness margin, not as the
primary nonlinearity mitigation strategy -- the items below address the root causes directly.

---

## Priority 1 -- Robustness to Nonlinearity and Disturbances

### 1.1 Online Sliding Hankel Window
**Why:** The offline Hankel matrix captures dynamics at one operating point. As conditions
change, the Hankel matrix must track the current regime.

- Replace the fixed offline dataset with a rolling buffer of the most recent `T_data` samples
- Re-build (or rank-1 update) the Hankel matrices after each control step
- Add a persistent excitation monitor -- warn or pause updates if the incoming data is
  insufficiently rich
- Tune the window length as a trade-off: too short loses diversity, too long retains
  stale dynamics

### 1.2 Disturbance Feedforward Estimator
**Why:** Persistent, correlated disturbances are not zero-mean noise.
`lambda_y` slack absorbs them as model mismatch instead of compensating for them.

- At each step, compute the residual between DeePC's predicted output and the actual
  measured output
- Fit a simple first-order disturbance model (e.g. constant or sinusoidal) to the
  residual history
- Inject the estimated disturbance as a known offset into the QP reference trajectory
- Validate against simulated persistent disturbances

### 1.3 Regime-Aware Regularization Scheduling
**Why:** A single `lambda_g` value is a blunt instrument. Different operating regimes
need different regularization strength.

- Define operating regimes by relevant state variables (e.g. speed, turning rate)
- Schedule `lambda_g` and `lambda_y` as lookup tables over the operating point
- Compare RMSE vs. fixed regularization across stress tests

---

### 1.4 Output Scaling / QP Conditioning (v3 candidate)
**Why:** Unscaled outputs (x grows to ~95m, y oscillates ±5m, v stays ~5) cause Yf cond~31k
and QP Hessian cond~823k. This is the root cause of K8, K10, K11, K12.

- Normalize each output channel by its std before building Hankel matrices
- Denormalize predicted outputs after solving
- Alternative: sparse QP form (keep u, y as separate decision variables)
- Expected: solve time drops to single-digit ms, optimal rate near 100%, L1 becomes viable

---

## Priority 2 -- Computational Performance

### 2.1 Sparse Hankel Exploitation
- Investigate sparsity structure in `U_p`, `U_f`, `Y_p`, `Y_f` blocks
- Pass explicit sparsity hints to the solver
- Profile whether Hankel construction or QP solve dominates runtime

### 2.2 Horizon Sensitivity Study
- Sweep `N` and `Tini` against RMSE and solve time
- Identify the Pareto-optimal (N, Tini) pair for each plant model
- Document the result in the version README

---

## Priority 3 -- Constraint Handling

### 3.1 Output Constraints
- Add convex spatial constraints on `y_f` (e.g. half-plane exclusion zones)
- Test with static obstacles placed on the reference path

### 3.2 Input Rate Constraints
- Add rate-of-change constraints on inputs (e.g. `u_f[k+1] - u_f[k]`)
- Physically necessary for actuator protection on real hardware

### 3.3 Soft vs. Hard Constraint Audit
- Document which constraints are hard (always satisfied) vs. soft (penalized violation)
- For real deployment, safety constraints should be hard; tracking should remain soft

### 3.4 Solver Failure Fallback
- Handle solver failures gracefully with a fallback control law (e.g. hold previous input,
  safe stop, or simple PID takeover)

---

## Priority 4 -- Plant Models

### 4.1 Additional Plant Models
- Add new plant models to `plants/` to validate DeePC generality:
  - Marine surface vessel (Fossen 3-DOF)
  - Quadrotor (simplified 6-DOF)
  - Process control system (e.g. CSTR, thermal system)
- Each plant should follow the same interface: `step(u) -> y`, `output`, `reset()`

### 4.2 Reference Trajectory Generators
- Replace the sinusoidal reference with configurable generators:
  - Waypoint-following with smooth interpolation
  - Step changes, ramps, figure-8 patterns
- Reference generators should be plant-agnostic

---

## Priority 5 -- Baselines and Benchmarking

### 5.1 PID Baseline
- Implement a decoupled PID controller
- Run identical stress suite scenarios
- Report RMSE, constraint violations, and compute time side-by-side with DeePC

### 5.2 MPC Baseline (Model-Based)
- Implement a linear MPC using the linearized plant model
- Establishes the ceiling: how much does knowing the model help?
- Gap between MPC and DeePC quantifies the cost of model-free operation

### 5.3 Automated Version Comparison Dashboard
- Extend `comparison/compare_versions.py` to include all metrics from the above baselines
- Generate a single HTML or PDF report with trajectory overlays, RMSE tables, and
  solve-time distributions

---

## Priority 6 -- Data Collection and Dataset Management

### 6.1 Persistent Excitation Verification (Quantitative)
- Compute and log the condition number of the Hankel matrix after data collection
- Warn if below a configurable threshold
- Expose this as a pre-flight check for hardware deployment

### 6.2 Dataset Versioning
- Save offline datasets as `.npz` with metadata (timestamp, excitation type, plant config)
- Allow the controller to load a pre-collected dataset instead of re-collecting at startup

### 6.3 Multi-Trajectory Dataset Merging
- Collect multiple short excitation trajectories and merge their Hankel contributions
- Improves coverage of the operating envelope without one very long excitation run

### 6.4 Investigate Effect of T_data on Tracking Quality
- Sweep `T_data` values and measure impact on RMSE and solver conditioning

---

## Priority 7 -- Code Quality

- [ ] Unit tests for Hankel construction, PE check, and plant models
- [ ] Integration test: run full simulation and assert tracking error bounds
- [ ] Type checking with mypy
- [ ] CI pipeline (GitHub Actions)
- [ ] Logging instead of print statements

---

## Future Extensions

- [ ] **DeePC-Hunt** -- hybrid DeePC formulation combining data-driven and model-based elements
- [ ] **Nonlinear DeePC variants** -- explore formulations that relax the LTI assumption directly
- [ ] **MIMO validation** -- test with higher-dimensional systems
- [ ] Real-time plotting / live simulation visualization
- [ ] Package distribution (pip-installable DeePC library)

---

## Known Issues

| ID | Description | Severity | Affects |
|----|-------------|----------|---------|
| K2 | Willems' lemma assumes LTI; bicycle model is nonlinear | Medium | v1_baseline |
| K3 | No disturbance compensation; persistent disturbances cause tracking bias | Medium | v1_baseline |
| K4 | T_data=200 is only ~1.9x the PE minimum; marginal for nonlinear regimes | Low | v1_baseline |
| K5 | ~~No rate constraints on inputs~~ — resolved in v2 | ~~Low~~ | ~~v1_baseline~~ |
| K7 | First few control steps show larger tracking error (warm-up transient from zero-input init) | Low | v1, v2 |
| K8 | Unscaled outputs cause poor Hankel conditioning (Yf cond~31k, QP Hessian cond~823k); OSQP hits max_iter on many steps | High | v1, v2 |
| K9 | High measurement noise (10x) degrades tracking beyond acceptable bounds; no robust DeePC formulation | Medium | v1, v2 |
| K10 | v2 solve time ~2x v1 (~760ms vs 325ms avg) due to additional constraints compounding K8 conditioning issue | High | v2 |
| K11 | L1 regularization unusable at current conditioning — 11% optimal solve rate, tracking degrades significantly. Needs output scaling first | Medium | v2 |
| K12 | Optimal solve rate dropped from 91% (v1) to 70% (v2 L2) due to larger constrained QP hitting OSQP max_iter | High | v2 |

---

## Completed

| Version | Item |
|---------|------|
| v1 | Core DeePC with L2 regularization |
| v1 | Parametric CVXPY QP with warm-starting |
| v1 | PRBS + multisine mixed excitation |
| v1 | 8-scenario stress test suite |
| v1 | Cross-version comparison infrastructure |
| v1 | Hidden heading state (realistic output model) |
| v1 | Kinematic bicycle plant model (decoupled in `plants/`) |
| v1 | Centralized config (single dataclass) |
| v1 | Sinusoidal reference tracking |
| v1 | Three-stage validation gate (A: 10/10, B: 9/10, C: 7/8) |
| v1 | Swapped CLARABEL -> OSQP for solver performance |
| v2 | Input rate constraints (hard) with cross-solve continuity |
| v2 | Soft output constraints via slack (sigma_out) |
| v2 | Configurable L1/L2 regularization on g and sigma_y |
| v2 | Hard/soft constraint audit formalized |
| v2 | L1 tested — unusable without conditioning fix (K11) |
