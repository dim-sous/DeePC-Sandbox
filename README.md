# DeePC Sandbox

A general-purpose Python platform for developing, testing, and benchmarking **Data-Enabled Predictive Control (DeePC)** algorithms. The controller uses pre-collected input-output data to predict and optimize future system behaviour through Hankel matrix representations — no explicit system model required.

The platform is evolving through **incremental, versioned upgrades** toward a reusable DeePC library. Plant models are decoupled from the controller and live in a shared `plants/` directory, making it straightforward to test DeePC against different dynamical systems.

---

## What This Platform Does

The platform answers the core question in data-driven control: **"Given only recorded input-output data from a system, can we build a controller that tracks a reference trajectory in real time — without ever identifying a model?"**

It does this through Data-Enabled Predictive Control (DeePC):

| Step | What Happens |
|------|-------------|
| **Offline data collection** | Drive the plant with persistently exciting inputs to capture system dynamics |
| **Hankel matrix construction** | Encode the collected data into block-Hankel matrices that implicitly represent all possible system trajectories |
| **Online optimization** | At each control step, solve a QP to find the optimal future input sequence that tracks the reference while respecting constraints |
| **Receding-horizon execution** | Apply only the first optimal input, measure the new output, and re-optimize |

This makes DeePC particularly suited for systems where accurate modeling is difficult, expensive, or impractical — the controller learns directly from data.

---

## Architecture

```
        Reference Trajectory
                |
                v
+----------------------------------------------+
|         DeePC Controller  (every Ts)         |
|                                              |
|  * Receding-horizon QP (CVXPY + OSQP)       |
|  * Hankel matrices from offline data         |
|  * Parametric problem (built once, warm-     |
|    started at each step)                     |
|  * Input constraints                         |
|  * Regularization (g penalty + output slack) |
|                                              |
|  Input: u_ini, y_ini (past Tini steps)       |
|         y_ref (next N steps)                 |
|  Output: u_optimal                           |
+-------------------+--------------------------+
                    |
                    v
+----------------------------------------------+
|       Plant Model  (every Ts)                |
|                                              |
|  Any dynamical system from plants/:          |
|  * Kinematic bicycle model (v1)              |
|  * (Future: marine vessels, quadrotors, ...) |
|                                              |
|  The controller never sees the model --      |
|  it only receives output measurements.       |
+----------------------------------------------+
```

### Why DeePC Instead of MPC?

Traditional Model Predictive Control (MPC) requires an explicit system model. DeePC replaces the model with a data-driven representation:

- **No system identification step**: The Hankel matrices encode the system's behaviour directly from raw input-output trajectories.
- **Handles nonlinearity implicitly**: While the theoretical guarantees are for LTI systems, regularization makes DeePC robust to mild nonlinearity and noise in practice.
- **Same computational structure as MPC**: The online optimization is a QP, solvable in milliseconds with standard solvers.

The trade-off is that DeePC requires sufficiently rich offline data (persistent excitation) and its performance degrades with severe nonlinearity or insufficient data.

---

## Project Structure

```
deepc-sandbox/
|-- pyproject.toml                 # Dependencies: cvxpy, numpy, matplotlib, scipy
|
|-- plants/                        # Shared plant models (decoupled from DeePC)
|   |-- __init__.py
|   +-- bicycle_model.py           #   Nonlinear kinematic bicycle model
|
|-- v1_baseline/                   # Version 1: baseline DeePC controller
|   |-- main.py                    #   Entry point with timing instrumentation
|   |-- stress_test.py             #   8-test stress suite with plots
|   |-- gate_v1.py                 #   Three-stage validation gate
|   |-- config/parameters.py       #   All tunable parameters (DeePCConfig dataclass)
|   |-- data/data_generation.py    #   PRBS + multisine excitation and data collection
|   |-- deepc/                     #   Core DeePC implementation
|   |   |-- deepc_controller.py    #     Parametric CVXPY QP with warm-starting
|   |   |-- hankel.py              #     Block-Hankel matrix construction
|   |   +-- regularization.py      #     Persistent excitation verification
|   +-- visualization/             #   Result plotting
|       +-- plot_results.py        #     Combined 6-panel figure
|
|-- comparison/                    # Cross-version comparison infrastructure
|   |-- metrics.py                 #   Metric computation from simulation results
|   |-- process_results.py         #   Load .npz files and produce metrics JSON
|   +-- compare_versions.py        #   Side-by-side table, CSV, and bar chart
|
|-- BACKLOG.md                     # Planned features and known issues
|
+-- results/                       # Simulation outputs (auto-generated)
    |-- v*_results.npz             #   Raw time series per version
    |-- v*_scalars.json            #   Scalar/list data per version
    |-- v*_metrics.json            #   Computed metrics per version
    |-- v*_results.png             #   6-panel result visualization
    |-- v*_stress_tests.png        #   Stress test visualizations
    |-- version_comparison.csv     #   All versions side-by-side
    +-- version_comparison.png     #   Comparison bar charts
```

Plant models are shared across all versions. Each version folder contains only DeePC-specific logic and imports plants from `plants/`.

---

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install and Run

```bash
cd deepc-sandbox
uv sync
```

Each version is independently runnable from the repository root:

```bash
# Run the baseline
uv run python -m v1_baseline.main

# Run stress tests
uv run python -m v1_baseline.stress_test

# Run the three-stage gate
uv run python -m v1_baseline.gate_v1
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

---

## Configuration

All parameters are centralized in each version's `config/parameters.py` via a `DeePCConfig` dataclass. See [v1_baseline/README.md](v1_baseline/README.md) for the full parameter reference.

---

## Plant Models

Plant models live in `plants/` and are decoupled from any DeePC configuration. Each plant accepts its own physical parameters directly, making them reusable across versions and experiments.

### Available Plants

| Plant | File | Description |
|-------|------|-------------|
| Kinematic bicycle | `plants/bicycle_model.py` | 2-D vehicle with steering and acceleration inputs. Nonlinear, discrete-time. |

Adding a new plant: create a class in `plants/` with `step(u) -> y`, `output`, and `reset()` methods. No DeePC dependency required.

---

## Version History

| Version | Key Addition |
|---------|-------------|
| **v1** (`v1_baseline`) | Core DeePC with L2 regularization, sinusoidal tracking, 8-test stress suite, 3-stage gate |
| v2 | _(planned)_ | See BACKLOG.md |

---

## Typical Results (v1, bicycle model)

- **Position RMSE**: ~0.36 m
- **Lateral RMSE**: ~0.16 m
- **Velocity RMSE**: ~0.23 m/s
- **Solver success rate**: 100% optimal solves
- **Constraint satisfaction**: All inputs within bounds

---

## Technical Stack

| Component | Library |
|-----------|---------|
| Optimization modeling | CVXPY |
| QP solver | OSQP (with SCS fallback) |
| Numerics | NumPy |
| Linear algebra | SciPy |
| Visualization | Matplotlib |

---

## References

- J. Coulson, J. Lygeros, F. Dorfler. *Data-Enabled Predictive Control: In the Shallows of the DeePC*. European Journal of Control, 2019.
- J.C. Willems, P. Rapisarda, I. Markovsky, B.L.M. De Moor. *A note on persistency of excitation*. Systems & Control Letters, 2005.
