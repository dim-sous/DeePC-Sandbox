# v1_baseline -- DeePC Baseline (Bicycle Model Plant)

Baseline DeePC controller using a kinematic bicycle model as the example plant. Implements the standard regularized DeePC formulation with L2 penalties on Hankel weights (g) and output slack (sigma_y), solving a parametric QP at each control step via CVXPY + OSQP.

## DeePC Formulation

At each time step k, the controller solves:

```
minimize    ||Y_f g - y_ref||^2_Q  +  ||U_f g||^2_R  +  lambda_g ||g||^2  +  lambda_y ||sigma_y||^2

subject to  U_p g  = u_ini                  (past input consistency -- hard)
            Y_p g  - y_ini = sigma_y        (past output consistency -- softened)
            u_lb  <=  U_f g  <=  u_ub       (input box constraints)
```

Where:
- `U_p, Y_p, U_f, Y_f` are partitioned Hankel matrices from offline data
- `g` is the Hankel combination weight vector
- `sigma_y` is a slack variable for output consistency (robustness to noise/nonlinearity)
- `u_ini, y_ini` are the most recent Tini input-output measurements
- `y_ref` is the reference trajectory over the N-step prediction horizon

### Key Parameters

| Parameter | Value | Role |
|-----------|-------|------|
| `Tini` | 3 | Past window -- balances initialization accuracy vs data requirements |
| `N` | 15 | Prediction horizon -- long enough for smooth tracking |
| `lambda_g` | 100.0 | Regularizes g toward zero, improving robustness to noisy Hankel data |
| `lambda_y` | 10000.0 | Heavily penalizes output slack, keeping past-data consistency tight |
| `Q_diag` | [10, 10, 1] | Prioritizes position tracking over velocity |
| `R_diag` | [0.1, 0.1] | Light input penalty -- allows aggressive control when needed |

## Plant Model

The plant is a **kinematic bicycle model** defined in `plants/bicycle_model.py`. The controller never sees this model -- it operates purely from data.

```
x(k+1)     = x(k)     + v(k) cos(theta(k)) Ts
y(k+1)     = y(k)     + v(k) sin(theta(k)) Ts
theta(k+1) = theta(k) + (v(k)/L_wb) tan(delta(k)) Ts
v(k+1)     = v(k)     + a(k) Ts

Output:  y = [x, y, v]    (heading theta is hidden)
Input:   u = [delta, a]   (steering, acceleration)
```

This is a nonlinear model with forward Euler integration. The nonlinearity creates inherent model mismatch that the DeePC regularization must handle.

## Module Structure

```
v1_baseline/
|-- main.py                   # Entry point: VERSION_TAG="v1_baseline"
|-- stress_test.py            # 8-test stress suite with plots
|-- gate_v1.py                # Three-stage validation gate
|-- config/
|   +-- parameters.py         # DeePCConfig dataclass with all tunable parameters
|-- data/
|   +-- data_generation.py    # PRBS, multisine generation + data collection loop
|-- deepc/
|   |-- deepc_controller.py   # Parametric CVXPY QP with warm-starting
|   |-- hankel.py             # Block-Hankel matrix construction + data partitioning
|   +-- regularization.py     # Persistent excitation rank check
+-- visualization/
    +-- plot_results.py       # 6-panel combined figure
```

Plant model imported from `plants/bicycle_model.py` (shared across versions).

## Running

```bash
# From repository root
uv run python -m v1_baseline.main

# Run stress tests
uv run python -m v1_baseline.stress_test

# Run three-stage gate
uv run python -m v1_baseline.gate_v1

# Compare with other versions
uv run python -m comparison.process_results
uv run python -m comparison.compare_versions
```

## Stress Tests

8 tests covering challenging conditions:

| # | Test | What It Validates | Pass Criteria |
|---|------|-------------------|---------------|
| 1 | High measurement noise (10x) | Robustness to noisy Hankel data | RMSE < 5 m, >50% optimal |
| 2 | Aggressive reference (2x freq, 2x amplitude) | Tracking under demanding trajectories | RMSE < 20 m, >30% optimal |
| 3 | Reduced dataset (T=50) | Graceful degradation with insufficient data | RMSE < 15 m, >20% optimal |
| 4 | Tight input constraints | Constraint satisfaction under limited actuation | All inputs within bounds |
| 5 | Nonlinear regime (high speed + large steering) | Performance deep in nonlinear territory | RMSE < 20 m, >30% optimal |
| 6 | Step reference change | Transient response to sudden offset | Converges within 2 m of target |
| 7 | Disturbance rejection | Recovery from external velocity perturbation | RMSE < 5 m |
| 8 | Long horizon (500 steps) | Sustained performance and solver reliability | RMSE < 3 m, max solve < 2 s |

## Results

| Metric | Value |
|--------|-------|
| Position RMSE | 0.36 m |
| Lateral RMSE | 0.16 m |
| Velocity RMSE | 0.23 m/s |
| Max position error | 0.92 m |
| Total control effort | 11.7 |
| Optimal solves | 100% |
| Mean slack norm | 0.007 |
