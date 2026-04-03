# DeePC Sandbox

A Python platform for **Data-Enabled Predictive Control (DeePC)** — a model-free control method that replaces system identification with raw input-output data. This project aims to apply DeePC to various plants, exploring the practical limits and engineering tradeoffs of data-driven predictive control.

## The Idea

Traditional model predictive control (MPC) requires an explicit system model. DeePC replaces it with a single offline dataset: apply persistently exciting inputs to the system, record what happens, and use matrix representation as an implicit model for online optimization. No physics equations, no parameter estimation — just data and a QP solver.

The theoretical foundation is **Willems' fundamental lemma**: for a linear time-invariant system, a sufficiently rich input-output trajectory contains all information needed to predict future behavior. DeePC builds on this to formulate a receding-horizon controller that solves a quadratic program at each time step.

## Architecture

```
run.py                  Experiment runner (data collection, simulation, HTML report)
tune.py                 Bayesian hyperparameter optimization (Optuna)

control/
    config.py           All tunable parameters in one dataclass
    controller.py       Sparse QP controller (CVXPY + OSQP)
    hankel.py           Block-Hankel matrix construction
    noise_estimator.py  Online prediction-residual noise estimation
    online_hankel.py    Sliding Hankel window for online adaptation
    regularization.py   Persistent excitation verification

plants/
    bicycle_model.py    Kinematic bicycle model + path error computation

sim/
    data_generation.py  Multi-episode stabilized data collection (PRBS + multisine)
    scenarios.py        Reference path generators
    simulation.py       Closed-loop simulation with error-based DeePC
```

## Quick Start

```bash
# Install
uv sync

# Run an experiment (60s simulation, generates HTML report)
uv run python run.py

# Customize parameters
uv run python run.py --Ts 0.039 --N 8 --Tini 4 --T-data 600 --lambda-g 32

# Bayesian parameter tuning (100 trials)
uv run python tune.py --n-trials 100 --sim-duration 10

# See all options
uv run python run.py --help
```

Reports are saved to `results/` as self-contained HTML files with interactive Plotly charts.

## DeePC Formulation

The controller solves at each time step:

```
minimize    (y - y_ref)' Q (y - y_ref) + u' R u + lambda_g ||g|| + lambda_y ||sigma_y||
subject to  [Up; Yp; Uf; Yf] g = [u_ini; y_ini + sigma_y; u; y]
            u_min <= u <= u_max
            |du| <= du_max
```

Where:
- `g` is the Hankel combination vector (selects a trajectory from the data)
- `sigma_y` is a slack variable for past-data consistency (handles noise/nonlinearity)
- `Up, Yp, Uf, Yf` are block-Hankel matrices from offline data
- The reference `y_ref` is always zero (minimize tracking errors)

## Key Results (Bicycle Model)

| Metric | Value |
|--------|-------|
| Lateral error RMSE | 0.54 m (60s simulation) |
| Heading error RMSE | 0.21 rad |
| Velocity error RMSE | 1.04 m/s |
| Solver | 100% optimal, ~150ms avg |
| Hankel conditioning | ~200 (error-based) vs ~55,000 (position-based) |

## Technical Stack

| Component | Library |
|-----------|---------|
| Optimization | CVXPY |
| QP solver | OSQP |
| Parameter tuning | Optuna (TPE sampler) |
| Visualization | Plotly |
| Numerics | NumPy, SciPy |

## References

- Coulson, Lygeros, Dorfler. *Data-Enabled Predictive Control: In the Shallows of the DeePC.* European Journal of Control, 2019.
- Willems, Rapisarda, Markovsky, De Moor. *A note on persistency of excitation.* Systems & Control Letters, 2005.
- Berberich, Koch, Scherer, Allgower. *Robust data-driven state-feedback design.* ACC, 2020.
