# DeePC Sandbox

A Python platform for **Data-Enabled Predictive Control (DeePC)** — a model-free control method that replaces system identification with raw input-output data. This project applies DeePC to various plant models, exploring the practical limits and engineering tradeoffs of data-driven predictive control.

## The Idea

Traditional model predictive control (MPC) requires an explicit system model. DeePC replaces it with a single offline dataset: apply persistently exciting inputs to the system, record what happens, and use matrix representation as an implicit model for online optimization. No physics equations, no parameter estimation — just data and a QP solver.

The theoretical foundation is **Willems' fundamental lemma**: for a linear time-invariant system, a sufficiently rich input-output trajectory contains all information needed to predict future behavior. DeePC builds on this to formulate a receding-horizon controller that solves a quadratic program at each time step.

## Architecture

```
run.py                  Experiment runner (data collection, simulation, HTML report)
tune.py                 Bayesian hyperparameter optimization (Optuna)

control/
    config.py           Algorithm parameters + build_deepc_config factory
    controller.py       Sparse QP controller (CVXPY + OSQP)
    hankel.py           Block-Hankel matrix construction
    noise_estimator.py  Online prediction-residual noise estimation
    online_hankel.py    Sliding Hankel window for online adaptation
    regularization.py   Persistent excitation verification

plants/
    base.py             PlantBase ABC + Constraints + DataCollectionConfig
    bicycle_model.py    Kinematic bicycle model (nonlinear, 2in/3out)
    coupled_masses.py   Coupled two-mass spring-damper (LTI MIMO, 2in/4out)

sim/
    data_generation.py  Multi-episode stabilized data collection (PRBS + multisine)
    scenarios.py        Reference scenario dispatcher
    simulation.py       Closed-loop simulation with error-based DeePC
```

## Quick Start

```bash
# Install
uv sync

# Run an experiment (default: bicycle model)
uv run python run.py

# Select a plant and scenario
uv run python run.py --plant bicycle --scenario lissajous
uv run python run.py --plant coupled_masses --scenario sinusoidal

# Customize algorithm parameters
uv run python run.py --Ts 0.039 --N 8 --Tini 4 --T-data 600 --lambda-g 32

# Run without constraints
uv run python run.py --plant coupled_masses --no-constraints

# Bayesian parameter tuning
uv run python tune.py --plant bicycle --n-trials 100 --sim-duration 10

# See all options
uv run python run.py --help
```

Reports are saved to `results/` as self-contained HTML files with interactive Plotly charts.

## Adding a New Plant

Implement `PlantBase` from `plants/base.py`:

```python
from plants.base import PlantBase, Constraints, DataCollectionConfig

class MyPlant(PlantBase):
    # Implement: m, p, Ts, input_names, output_names, state
    # Implement: step(), reset(), get_output(), get_constraints()
    # Implement: get_default_config_overrides(), get_scenarios()
    # Implement: get_data_collection_config(), make_episode_initial_state()
    # Optional:  plot_training_data(), plot_simulation_results(), compute_custom_metrics()
    ...
```

Then add it to `PLANT_REGISTRY` in `run.py` and run with `--plant my_plant`.

## Available Plants

| Plant | Type | Inputs | Outputs | Scenarios |
|-------|------|--------|---------|-----------|
| `bicycle` | Nonlinear | steering, acceleration | lateral/heading/velocity errors | default (sinusoidal), lissajous |
| `coupled_masses` | LTI MIMO | F1, F2 (forces) | position/velocity errors (x2) | default (step sequence), sinusoidal |

## DeePC Formulation

The controller solves at each time step:

```
minimize    (y - y_ref)' Q (y - y_ref) + u' R u + lambda_g ||g|| + lambda_y ||sigma_y||
subject to  [Up; Yp; Uf; Yf] g = [u_ini; y_ini + sigma_y; u; y]
            u_min <= u <= u_max
            du_min <= du <= du_max
```

Where:
- `g` is the Hankel combination vector (selects a trajectory from the data)
- `sigma_y` is a slack variable for past-data consistency (handles noise/nonlinearity)
- `Up, Yp, Uf, Yf` are block-Hankel matrices from offline data
- The reference `y_ref` is always zero (minimize tracking errors)

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
