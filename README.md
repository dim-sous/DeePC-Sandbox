# DeePC Sandbox

A personal research sandbox for **Data-Enabled Predictive Control (DeePC)** — a model-free control method that replaces system identification with raw input-output data. It applies DeePC to a couple of plant models to explore the practical behavior and engineering tradeoffs of data-driven predictive control. This is an active personal project, not a library or product.

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
    noise_estimator.py  Online prediction-residual noise estimation (optional, --noise-adaptive)
    online_hankel.py    Sliding Hankel window for online adaptation (optional, --online-hankel)
    regularization.py   Persistent excitation verification

plants/
    base.py             PlantBase ABC + Constraints + DataCollectionConfig
    bicycle_model.py    Kinematic bicycle model (nonlinear, 2in/3out)
    coupled_masses.py   Coupled two-mass spring-damper (LTI MIMO, 2in/4out)

sim/
    data_generation.py  Multi-episode stabilized data collection (PRBS + multisine)
    scenarios.py        Reference scenario dispatcher
    simulation.py       Closed-loop simulation with error-based DeePC

tests/
    reference/          The safety net: today's numbers, frozen (cases.py, make.py, *.npz)
    test_reference.py   Today's code against the frozen numbers, bit for bit
    test_conventions.py CLAUDE.md's heft budget

docs/project/
    failures.md         The incidents the rules in CLAUDE.md were written after (D1-D8)
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

# Customize algorithm parameters (--Ts does not reach the plant: docs/project/failures.md, D1)
uv run python run.py --Ts 0.039 --N 8 --Tini 4 --T-data 600 --lambda-g 32

# Run without constraints
uv run python run.py --plant coupled_masses --no-constraints

# Optional / experimental features (both off by default)
uv run python run.py --online-hankel    # slide the Hankel window using closed-loop data
uv run python run.py --noise-adaptive   # scale regularization from estimated noise

# Bayesian parameter tuning
# (set aside: its Ts search runs through D1, docs/project/failures.md)
uv run python tune.py --plant bicycle --n-trials 100 --sim-duration 10

# See all options
uv run python run.py --help

# The test suite: CLAUDE.md's heft budget and the safety net
uv run pytest
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

Input and input-rate constraints are hard. Output constraints, when enabled, are enforced softly via an extra slack term (`lambda_out`), omitted above for clarity.

## Technical Stack

| Component | Library |
|-----------|---------|
| Optimization | CVXPY |
| QP solver | OSQP |
| Parameter tuning | Optuna (TPE sampler) |
| Visualization | Plotly |
| Numerics | NumPy, SciPy |

## Claims from before the rebuild, not re-measured

Written into the old CLAUDE.md before any run was recorded. No record backs them; each stays
unverified until one re-measures it.

- Absolute or monotonic outputs destroy Hankel conditioning; error-based or shift-invariant
  outputs are used instead.
- Open-loop data collection produces drifted, asymmetric output distributions; a stabilising
  baseline during collection keeps the errors centred.
- Tuning on short simulations and evaluating on long ones exposes drift.
- Sampling time (Ts) dominates tracking accuracy more than any regularisation parameter. The
  runs behind it went through `--Ts`, which never reaches the plant (`docs/project/failures.md`,
  D1).
- One-at-a-time parameter sweeps miss interactions; a Bayesian joint search finds better
  configurations.
- Training data quality matters more than algorithm complexity: a well-conditioned Hankel
  matrix with good coverage beats any amount of regularisation tuning.

## References

- Coulson, Lygeros, Dorfler. *Data-Enabled Predictive Control: In the Shallows of the DeePC.* European Journal of Control, 2019.
- Willems, Rapisarda, Markovsky, De Moor. *A note on persistency of excitation.* Systems & Control Letters, 2005.
- Berberich, Koch, Scherer, Allgower. *Robust data-driven state-feedback design.* ACC, 2020.
