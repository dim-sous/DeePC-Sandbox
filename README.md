# DeePC Autonomous Vehicle Control Platform

A complete Python platform for **data-driven predictive control** of an autonomous vehicle using only input-output data — no explicit system model required. The controller uses pre-collected driving data to predict and optimize future vehicle behaviour through Hankel matrix representations, completely bypassing system identification or model derivation.

This is the same class of data-driven control algorithms being evaluated for autonomous driving, robotics, and industrial process control. The platform is evolving through **incremental, versioned upgrades** toward a production-grade controller — each version is independently runnable with quantitative metrics and version-to-version comparison.

---

## What This Platform Does

The platform answers the core question in data-driven control: **"Given only recorded input-output data from a system, can we build a controller that tracks a reference trajectory in real time — without ever identifying a model?"**

It does this through Data-Enabled Predictive Control (DeePC):

| Step | What Happens |
|------|-------------|
| **Offline data collection** | Drive the vehicle with persistently exciting inputs (PRBS + multisine) to capture system dynamics |
| **Hankel matrix construction** | Encode the collected data into block-Hankel matrices that implicitly represent all possible system trajectories |
| **Online optimization** | At each control step, solve a QP to find the optimal future input sequence that tracks the reference while respecting constraints |
| **Receding-horizon execution** | Apply only the first optimal input, measure the new output, and re-optimize |

This makes DeePC particularly suited for systems where accurate modeling is difficult, expensive, or impractical — the controller learns directly from data.

---

## Architecture

```
        Reference Trajectory
        (sinusoidal path at v_ref)
                │
                ▼
┌──────────────────────────────────────────────┐
│         DeePC Controller  (every Ts)         │
│                                              │
│  • Receding-horizon QP (CVXPY + CLARABEL)    │
│  • Hankel matrices from offline data         │
│  • Parametric problem (built once, warm-     │
│    started at each step)                     │
│  • Input constraints (steering, accel)       │
│  • Regularization (g penalty + output slack) │
│                                              │
│  Input: u_ini, y_ini (past Tini steps)       │
│         y_ref (next N steps)                 │
│  Output: u_optimal (steering, acceleration)  │
└───────────────────┬──────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│       Vehicle Plant  (every Ts = 0.1 s)      │
│                                              │
│  • Nonlinear kinematic bicycle model         │
│  • State: [x, y, theta, v]                  │
│  • Input: [steering, acceleration]           │
│  • Output: [x, y, v] (heading is hidden)     │
│  • Forward Euler integration                 │
│                                              │
│  The controller never sees the model —       │
│  it only receives output measurements.       │
└──────────────────────────────────────────────┘
```

### Why DeePC Instead of MPC?

Traditional Model Predictive Control (MPC) requires an explicit system model — either derived from first principles or identified from data. DeePC replaces the model with a data-driven representation:

- **No system identification step**: The Hankel matrices encode the system's behaviour directly from raw input-output trajectories. No transfer function, state-space model, or neural network is needed.
- **Handles nonlinearity implicitly**: While the theoretical guarantees are for LTI systems, regularization (lambda_g, lambda_y) makes DeePC robust to mild nonlinearity and noise in practice.
- **Same computational structure as MPC**: The online optimization is a QP, solvable in milliseconds with standard solvers. The receding-horizon principle is identical.

The trade-off is that DeePC requires sufficiently rich offline data (persistent excitation) and its performance degrades with severe nonlinearity or insufficient data.

---

## Project Structure

```
deepc_project/
├── pyproject.toml                 # Dependencies: cvxpy, numpy, matplotlib, scipy
│
├── v1_baseline/                   # Version 1: baseline DeePC controller
│   ├── main.py                    #   Entry point with timing instrumentation
│   ├── stress_test.py             #   8-test stress suite with plots
│   ├── config/parameters.py       #   All tunable parameters (DeePCConfig dataclass)
│   ├── data/data_generation.py    #   PRBS + multisine excitation and data collection
│   ├── deepc/                     #   Core DeePC implementation
│   │   ├── deepc_controller.py    #     Parametric CVXPY QP with warm-starting
│   │   ├── hankel.py              #     Block-Hankel matrix construction
│   │   └── regularization.py      #     Persistent excitation verification
│   ├── simulation/                #   Plant simulator
│   │   └── vehicle_simulator.py   #     Nonlinear kinematic bicycle model
│   └── visualization/             #   Result plotting
│       └── plot_results.py        #     Combined 6-panel figure
│
├── comparison/                    # Cross-version comparison infrastructure
│   ├── metrics.py                 #   Metric computation from simulation results
│   ├── process_results.py         #   Load .npz files and produce metrics JSON
│   └── compare_versions.py        #   Side-by-side table, CSV, and bar chart
│
├── BACKLOG.md                     # Planned features and known issues
│
└── results/                       # Simulation outputs (auto-generated)
    ├── v*_results.npz             #   Raw time series per version
    ├── v*_scalars.json            #   Scalar/list data per version
    ├── v*_metrics.json            #   Computed metrics per version
    ├── v*_results.png             #   6-panel result visualization
    ├── v*_stress_tests.png        #   Stress test visualizations
    ├── version_comparison.csv     #   All versions side-by-side
    └── version_comparison.png     #   Comparison bar charts
```

Each version is fully self-contained with its own `README.md` documenting changes from its predecessor.

---

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install and Run

```bash
cd deepc_project
uv sync
```

Each version is independently runnable from the repository root:

```bash
# Run the baseline
uv run python -m v1_baseline.main

# Run stress tests
uv run python -m v1_baseline.stress_test
```

Results (`.npz` time series, `.json` metrics, `.png` plots) are saved to `results/`.

### Comparing Versions

After running one or more versions, generate a side-by-side comparison:

```bash
# Compute metrics from saved .npz files
uv run python -m comparison.process_results

# Print comparison table, save CSV and bar chart
uv run python -m comparison.compare_versions
```

### Expected Output

Each simulation takes approximately 1 minute (150 control steps with QP solves). Console output shows:

```
Collecting persistently exciting data...
  Data collected: u (200, 2), y (200, 3)
Building DeePC controller...
  Controller ready.
Running closed-loop simulation (150 steps)...
  Step    1/150  status=optimal  cost=109.49  solve=537.9ms
  Step   50/150  status=optimal  cost=37.64   solve=375.7ms
  Step  150/150  status=optimal  cost=30.39   solve=429.4ms

Done. Optimal solves: 150/150  (100%)
```

---

## Configuration

All parameters are centralized in each version's `config/parameters.py` via the `DeePCConfig` dataclass:

### Vehicle

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Ts` | 0.1 s | Sampling time |
| `L_wheelbase` | 2.5 m | Vehicle wheelbase |
| `v_ref` | 5.0 m/s | Reference forward velocity |

### Data Collection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T_data` | 200 | Offline data samples |
| `noise_std_output` | 0.01 | Measurement noise standard deviation |
| `input_amplitude_delta` | 0.3 rad | Steering excitation amplitude |
| `input_amplitude_a` | 1.0 m/s^2 | Acceleration excitation amplitude |

### DeePC Horizons and Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Tini` | 3 | Past window length |
| `N` | 15 | Prediction horizon |
| `Q_diag` | [10, 10, 1] | Output tracking weights [x, y, v] |
| `R_diag` | [0.1, 0.1] | Input cost weights [steering, acceleration] |
| `lambda_g` | 100.0 | Hankel weight regularization |
| `lambda_y` | 10000.0 | Output slack penalty |

### Input Constraints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delta_max` | 0.5 rad | Max steering angle |
| `a_max` | 3.0 m/s^2 | Max acceleration |
| `a_min` | -5.0 m/s^2 | Max braking deceleration |

---

## Output Visualization

The platform generates a single 6-panel figure showing:

| Panel | What It Shows | Why It Matters |
|-------|--------------|----------------|
| **Trajectory (x-y)** | 2-D vehicle path vs sinusoidal reference | Overall tracking quality at a glance |
| **Velocity tracking** | v_ref vs actual velocity over time | Validates speed regulation alongside lateral tracking |
| **Longitudinal tracking** | x_ref vs actual x over time | Shows forward position accuracy |
| **Lateral tracking** | y_ref vs actual y over time | Shows the most challenging axis — lateral control through a nonlinear plant |
| **Control inputs** | Steering and acceleration on dual y-axes with constraint bounds | Verifies inputs stay within limits and shows control effort |

---

## Solver Technology

The DeePC optimization is formulated using **CVXPY** and solved with the **CLARABEL** interior-point solver (with SCS as fallback).

Key solver features used:
- **Parametric problem construction**: The QP is built once using `cp.Parameter` objects. At each control step, only parameter values are updated — avoiding repeated compilation overhead.
- **Warm-starting**: Previous solutions seed the next solve, reducing computation time.
- **Diagonal weight matrices**: Tracking and input costs use `cp.multiply` with weight vectors instead of `cp.quad_form`, which is significantly faster for diagonal weights.
- **Soft output constraints**: Past-data consistency is enforced via slack variables (sigma_y) with heavy L2 penalty, guaranteeing feasibility even under noise and model mismatch.

---

## Design Decisions

### Regularization Over Hard Constraints
The past output consistency equation `Y_p * g = y_ini` is softened to `Y_p * g - y_ini = sigma_y` with a heavy penalty on sigma_y. This is essential for robustness — with noisy data and a nonlinear plant, hard equality on past data would frequently make the QP infeasible.

### Mixed Excitation Signals
Data collection uses a 50/50 blend of structured signals (PRBS for steering, multisine for acceleration) and uniform random noise. The structured component provides rich spectral content, while the random component fills gaps and strengthens the persistent excitation condition.

### Hidden Heading State
The vehicle heading (theta) is intentionally excluded from the output vector. The controller must infer turning behaviour from position and velocity changes alone — a realistic constraint since heading is often poorly measured or unavailable in practice.

### Receding Horizon with Zero-Input Warm-Up
The first Tini steps use zero inputs to fill the past buffers before the controller starts. This creates a brief transient but avoids the need for artificial initial trajectory data.

---

## Version Upgrades

The platform evolves through incremental, independently runnable upgrades. Each version adds one major capability while preserving everything from previous versions.

| Version | Name | Key Addition |
|---------|------|-------------|
| **v1** | `v1_baseline` | Core DeePC with L2 regularization, sinusoidal tracking, 8-test stress suite |
| v2 | _(planned)_ | See BACKLOG.md for candidate upgrade directions |

### Candidate Upgrade Directions

- Regularized DeePC formulations (1-norm, elastic net)
- Improved handling of noisy measurements
- Better Hankel matrix conditioning
- Constraint handling improvements (output constraints, rate limits)
- Dataset management and online updating
- Computational improvements for real-time use
- Robustness to insufficient or poorly excited datasets
- Comparison against PID or MPC baselines

---

## Typical Results

For the default configuration (200 data samples, 150 control steps, sinusoidal reference):

- **Position RMSE**: ~0.36 m
- **Lateral RMSE**: ~0.16 m
- **Velocity RMSE**: ~0.23 m/s
- **Solver success rate**: 100% optimal solves
- **Average solve time**: ~390 ms per step
- **Constraint satisfaction**: All inputs within bounds

---

## Technical Stack

| Component | Library | Version |
|-----------|---------|---------|
| Optimization modeling | CVXPY | >= 1.8.1 |
| QP solver | CLARABEL (with SCS fallback) | bundled |
| Numerics | NumPy | >= 2.4.2 |
| Linear algebra | SciPy | >= 1.17.1 |
| Visualization | Matplotlib | >= 3.10.8 |

---

## References

- J. Coulson, J. Lygeros, F. Dorfler. *Data-Enabled Predictive Control: In the Shallows of the DeePC*. European Journal of Control, 2019.
- J.C. Willems, P. Rapisarda, I. Markovsky, B.L.M. De Moor. *A note on persistency of excitation*. Systems & Control Letters, 2005.
