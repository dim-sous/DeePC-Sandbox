# DeePC Sandbox -- Backlog

Items are grouped by theme and ordered by priority within each group.
Regularization (`lambda_g`, `lambda_y`) is treated as a robustness margin, not as the
primary nonlinearity mitigation strategy -- the items below address the root causes directly.

---

## Priority 1 -- Robustness to Nonlinearity and Disturbances

### ~~1.1 Online Sliding Hankel Window~~ — implemented in v6
v6 adds append-then-slide Hankel window with cp.Parameter matrices (no QP rebuild).
Hankel blocks (Up, Yp, Uf, Yf) updated each step with closed-loop I/O data.
Quality filter: only adds data when prediction residual < 1.0m.

**Key finding:** Online Hankel degrades nominal tracking (RMSE 0.66→1.04m at warmup=50)
because closed-loop data contains tracking errors that bias the implicit model. With
warmup=100 (default), nominal passes but stress tests are too short (150 steps) for
online adaptation to help. The mechanism works structurally (A13/A14 validated) but
needs longer episodes or better data quality filtering to outperform the offline baseline.
Net result: same C-stage pass rate as v5 (4/9) with no regression.

### 1.2 Disturbance Feedforward Estimator
**Why:** Persistent, correlated disturbances are not zero-mean noise.
`lambda_y` slack absorbs them as model mismatch instead of compensating for them.

- At each step, compute the residual between DeePC's predicted output and the actual
  measured output
- Fit a simple first-order disturbance model (e.g. constant or sinusoidal) to the
  residual history
- Inject the estimated disturbance as a known offset into the QP reference trajectory
- Validate against simulated persistent disturbances

### ~~1.3 Regime-Aware Regularization Scheduling~~ — resolved in v5
Implemented as noise-adaptive regularization: lambda_g and lambda_y scale online
proportional to estimated prediction residual variance (Wasserstein ball radius
per Coulson/Lygeros/Dörfler arXiv:1903.06804). Cuts C2 RMSE by 47% (23.2 → 12.2m)
but cannot compensate for fundamentally corrupted or insufficient Hankel data.

---

### 1.4 QP Conditioning — Sparse Form (v4 candidate)
**Why:** The condensed QP form (g as sole decision variable) produces a Hessian
`Yf'QYf + Uf'RUf + lambda_g I` that inherits Yf's poor conditioning (cond~823k).
v3's std-scaling fixed solve time (24ms, 100% optimal) but degraded position tracking
(RMSE 0.36 → 0.92) because scaling distorts the Q weighting — x_std=27 makes
x-errors invisible to the optimizer. **Scaling solves the wrong problem.**

The original DeePC paper (Coulson, Lygeros, Dörfler 2019) formulates the QP with
u and y as explicit decision variables, not substituted out. This "sparse form" keeps
the Hessian block-diagonal (Q, R, lambda_g I) — inherently well-conditioned regardless
of output scale. It also enables L1 norms as the paper recommends (L1 on g and sigma_y).

Proposed approach for v4:
- Sparse QP: decision variables are (g, u, y, sigma_y, sigma_out) with equality
  constraints u = Uf g, y = Yf g linking them
- L1 norms on g and sigma_y (per original paper, eq. 8)
- Low-rank SVD approximation of Hankel matrix (paper's third regularization)
- Remove output scaling (no longer needed with sparse form)
- Expected: good conditioning WITHOUT distorting tracking weights

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
| K7 | Startup transient: first ~3s show 0.5-1.0m tracking error from zero-input buffer initialization. Reference blending (v4) reduces peak from 1.0 to 0.55m but does not eliminate it. P-controller warm-up and extended DeePC warm-up both made it worse. Root cause: zero-input buffers are inconsistent with Hankel data. Needs structural fix (e.g. replay training data segment through sim) | Medium | all |
| K8 | ~~Poor QP conditioning from condensed form~~ — resolved in v4 (sparse QP form, block-diagonal Hessian) | ~~High~~ | ~~v1, v2, v3~~ |
| K9 | High measurement noise (10x) degrades tracking. v5 adds noise-adaptive regularization (Wasserstein ball scaling) but the root cause is corrupted Hankel training data, not online estimation. RMSE=36m under 10x noise — regularization cannot fix a fundamentally wrong implicit model. Needs data denoising or sliding Hankel window | High | all |
| K10 | ~~v2 solve time regression~~ — resolved in v4 (7ms avg) | ~~High~~ | ~~v2, v3~~ |
| K11 | ~~L1 unreliable in condensed form~~ — resolved in v4 (sparse form, L2/L1 default, 100% optimal) | ~~Medium~~ | ~~v2, v3~~ |
| K12 | ~~Low optimal solve rate~~ — resolved in v4 (100% optimal) | ~~High~~ | ~~v1, v2~~ |
| K13 | Tracking degrades when simulation visits speed regimes outside training data. Error accumulates and persists even after returning to known regime | High | v3 |
| K14 | Increasing T_data alone does not improve tracking; wider excitation amplitude is needed for data coverage | Medium | v3 |
| K15 | Nonlinear regime (v=10) stress test fails — bicycle model dynamics at high speed are far from LTI assumption | Medium | v3, v4 |
| K16 | v3 output scaling distorts Q weighting — position RMSE regressed from 0.36 (v1) to 0.92 (v3). Fixed in v4 by using sparse form instead of scaling | ~~High~~ | ~~v3~~ |
| K17 | L1 on g causes erratic control signals (bang-bang-like input switching). L2 on g with L1 on sigma_y (v4 default) gives smooth control | Low | v4 |
| K18 | C1 (high noise) stress test fails — v5 adds noise-adaptive regularization but RMSE=36m persists. Root cause: 10x noise corrupts the offline Hankel matrix itself. Online lambda scaling reaches cap (10x) by step 6 but prediction residuals compound tracking error into the noise estimate. Needs structural fix: data denoising before Hankel construction or sliding Hankel window | High | all |
| K19 | C4 (tight constraints) stress test fails — delta_max=0.1 with rate limits d_delta=0.1 leaves zero maneuvering room. RMSE=3.2m, trajectory barely tracks. Constraint satisfaction also violated at solver tolerance boundary. Not a regularization issue — fundamental actuation limit | Medium | v4, v5 |
| K20 | C2/C9 (aggressive reference) stress tests fail — aggressive sinusoidal (amp=10, freq=0.1) operates far outside the linear regime covered by training data. v5 adaptive lambda improves C2 RMSE 23.2→12.2m (47% reduction) but 12.2m is still not production-grade tracking. Fundamental: offline Hankel matrix does not contain information about these operating regimes | High | all |
| K21 | Wider excitation amplitude in data collection (δ=0.5 vs 0.3) does not improve tracking on hard scenarios — pushes plant into nonlinear regimes during training, degrading Hankel matrix quality for the linear assumption | Low | v4 |
| K22 | v5 noise estimator conflates measurement noise with tracking error. Prediction residuals grow to 3.4m under high noise (vs 0.1m actual noise) because tracking error dominates. A separate noise/mismatch decomposition would improve the Wasserstein ball calibration | Medium | v5 |
| K23 | v5 gate thresholds tightened to production-grade (RMSE<2m for most C-tests, opt>90%, solve<0.5s). All 5 C-stage failures trace to the same root cause: offline Hankel matrix does not represent the operating regime (noise, data scarcity, nonlinearity, constraint mismatch) | — | v5 |
| K24 | Online Hankel (v6) degrades nominal tracking: closed-loop data contains tracking errors that bias the implicit model. RMSE 0.66→1.04m at warmup=50. Warmup=100 preserves nominal but limits online adaptation to last 50 steps of 150-step stress tests — insufficient for meaningful improvement | Medium | v6 |
| K25 | 150-step stress tests are too short for online Hankel adaptation. The mechanism needs ~200+ steps of good-quality closed-loop data to meaningfully improve the implicit model. Stress test duration should scale with hankel_warmup_steps | Low | v6 |

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
| v3 | Output scaling (y_std normalization before Hankel construction) |
| v3 | QP Hessian conditioning 823k → 16k, solve time 760ms → 24ms |
| v3 | 100% optimal solve rate across all L2 configurations |
| v3 | Phased data generation (PRBS + chirp + multisine) for wider regime coverage |
| v3 | 8-phase challenging reference (lane changes, speed sweeps, slalom, braking) |
| v3 | Horizon/parameter sweep: Tini=3, N=15, lambda_g=10, a_amp=3.0 confirmed optimal |
| v3 | Gate: A 13/13, B 12/12, C 7/9 in 71s (v2: 725s) |
| v4 | Sparse QP form per original DeePC paper (u, y as explicit decision variables) |
| v4 | L2 on g + L1 on sigma_y (paper-recommended, smooth control) |
| v4 | Block-diagonal Hessian — conditioning independent of output scale |
| v4 | Best tracking: pos RMSE 0.288 (v1: 0.358), 7ms solve, 100% optimal |
| v4 | sigma_y near-zero (L1 exact penalty working as designed) |
| v4 | Reference blending for smooth startup (partial fix for K7) |
| v4 | Parameter sweep: lambda_g=5, L2/L1 norms, d_delta=0.1, da=0.5 confirmed |
| v5 | Noise-adaptive regularization per Coulson/Lygeros/Dörfler (arXiv:1903.06804) |
| v5 | lambda_g, lambda_y as cp.Parameter — updated online based on prediction residual variance |
| v5 | NoiseEstimator: rolling window residual variance, Wasserstein ball radius scaling |
| v5 | Output constraint tightening proportional to estimated noise |
| v5 | C2 RMSE improved 47% (23.2→12.2m) via adaptive regularization |
| v5 | Production-grade gate thresholds: RMSE<2m, opt>90%, solve<0.5s |
| v5 | Gate: A 15/15, B 15/15, C 4/9 — 5 failures all trace to LTI assumption (K9/K18/K19/K20) |
| v6 | Online sliding Hankel window (append-then-slide) per arXiv:2407.16066 |
| v6 | Hankel blocks as cp.Parameter matrices — no QP rebuild on update, warm-start preserved |
| v6 | Quality filter: only adds closed-loop data when prediction residual < 1.0m |
| v6 | Warmup period (default 100 steps) to prevent early tracking errors from corrupting Hankel |
| v6 | Gate: A 17/17, B 15/15, C 4/9 — same C-stage as v5, no regression |
