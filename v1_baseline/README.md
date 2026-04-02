# v1_baseline -- DeePC Baseline (Bicycle Model)

Proof-of-concept DeePC controller. Solves a parametric QP at each step via CVXPY + OSQP.

## Formulation

### Data Matrices

From offline input-output data `(u_data, y_data)` of length `T_data`, build block-Hankel matrices of depth `L = Tini + N`, then partition into past/future blocks:

```
H_L(u) -> U_p (Tini*m rows),  U_f (N*m rows)
H_L(y) -> Y_p (Tini*p rows),  Y_f (N*p rows)
```

Columns: `T_data - L + 1`. Requires persistent excitation: `rank(H_L(u)) >= L*m`.

### QP (condensed form)

Decision variables: `g` (column combination weights), `sigma_y` (output slack).

```
min   ||Y_f g - y_ref||^2_Q  +  ||U_f g||^2_R  +  lambda_g ||g||^2  +  lambda_y ||sigma_y||^2

s.t.  U_p g  = u_ini                   (hard)
      Y_p g  - y_ini = sigma_y         (soft via slack)
      u_lb  <=  U_f g  <=  u_ub        (input box constraints)
```

Receding horizon: solve, apply first input `u* = (U_f g*)[0]`, repeat.

**Note:** This is the condensed form — `u` and `y` are substituted out via `u = U_f g`, `y = Y_f g`. The Hessian `Y_f^T Q Y_f + U_f^T R U_f + lambda_g I` inherits the conditioning of `Y_f`. See Known Issues K8.

### Plant

Kinematic bicycle (nonlinear, forward Euler). The controller never sees the model.

```
State:   [x, y, theta, v]
Input:   [delta, a]       (steering, acceleration)
Output:  [x, y, v]        (heading hidden)
```

### Parameters

| Parameter | Value | Role |
|-----------|-------|------|
| Tini | 3 | Past window |
| N | 15 | Prediction horizon |
| lambda_g | 100 | L2 on g |
| lambda_y | 10000 | L2 on slack |
| Q_diag | [10, 10, 1] | Position > velocity tracking |
| R_diag | [0.1, 0.1] | Light input penalty |

## Running

```bash
uv run python -m v1_baseline.main        # baseline simulation
uv run python -m v1_baseline.gate         # three-stage gate
```

## Gate Results

| Stage | Result |
|-------|--------|
| A — Validation | 10/10 |
| B — Evaluation | 9/10 (optimal solve rate 91%, threshold >= 80%) |
| C — Stress Test | 7/8 (high noise fails — known limitation K9) |

## Known Limitations

- **K8**: Unscaled outputs cause poor QP conditioning (Hessian cond ~823k). OSQP hits max_iter on many steps.
- **K9**: 10x measurement noise degrades tracking beyond bounds. No robust DeePC formulation.
- See BACKLOG.md for full list.
