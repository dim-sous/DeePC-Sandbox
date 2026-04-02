# DeePC Sandbox

A general-purpose Python platform for developing, testing, and benchmarking **Data-Enabled Predictive Control (DeePC)** algorithms. The controller uses pre-collected input-output data to predict and optimize future system behaviour through Hankel matrix representations — no explicit system model required.

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

---

## Project Structure

```
deepc/
├── control/                       # DeePC control logic (pure algorithm)
│   ├── config.py                  #   DeePCConfig dataclass (all tunable parameters)
│   ├── controller.py              #   Sparse QP controller with online Hankel + noise-adaptive params
│   ├── hankel.py                  #   Block-Hankel matrix construction
│   ├── regularization.py          #   Persistent excitation verification
│   ├── noise_estimator.py         #   Rolling prediction-residual noise estimator
│   └── online_hankel.py           #   Sliding Hankel window (append-then-slide)
│
├── sim/                           # Simulation, evaluation, comparison
│   ├── __main__.py                #   CLI entry point
│   ├── gate.py                    #   Gate entry point
│   ├── eval/
│   │   ├── scenarios.py           #     Reference trajectory generators
│   │   ├── simulation.py          #     Closed-loop simulation runner
│   │   ├── data_generation.py     #     PRBS + multisine data collection
│   │   ├── visualization.py       #     Result plotting
│   │   └── gate.py                #     Three-stage gate (A/B/C)
│   └── comparison/
│       ├── metrics.py             #     Metric computation
│       ├── stress_configs.py      #     Shared stress test scenarios
│       ├── process_results.py     #     Load results and compute metrics
│       └── compare_versions.py    #     Side-by-side table, CSV, bar chart
│
├── plants/                        # Shared plant models
│   └── bicycle_model.py           #   Nonlinear kinematic bicycle model
│
└── results/                       # Simulation outputs (auto-generated)
```

---

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install and Run

```bash
uv sync
```

### Running Simulations

```bash
# Base DeePC (sparse QP form)
uv run python -m sim

# With noise-adaptive regularization
uv run python -m sim --noise-adaptive

# With online sliding Hankel window
uv run python -m sim --online-hankel

# Both features
uv run python -m sim --noise-adaptive --online-hankel

# With scenario selection
uv run python -m sim --scenario hard
uv run python -m sim --scenario circle
uv run python -m sim --scenario square
uv run python -m sim --scenario zigzag
```

### Running the Gate

```bash
# Run all feature combos (base, na, oh, na+oh)
uv run python -m sim.gate

# Run a specific combo
uv run python -m sim.gate --combo base
uv run python -m sim.gate --combo na+oh
```

### Comparing Results

```bash
uv run python -m sim.comparison.process_results
uv run python -m sim.comparison.compare_versions
```

Results are saved to `results/<tag>/` where tag is `base`, `na`, `oh`, `na+oh` (with optional scenario suffix like `na_hard`).

---

## Features

The platform uses a single controller class with optional features enabled via CLI flags:

| Flag | Feature | Origin |
|------|---------|--------|
| *(base)* | Sparse QP form with L2/L1 regularization, input rate constraints, soft output constraints | — |
| `--noise-adaptive` | Online noise estimation with adaptive lambda scaling and constraint tightening | Coulson/Lygeros/Dörfler (arXiv:1903.06804) |
| `--online-hankel` | Sliding Hankel window that adapts implicit model with closed-loop data | arXiv:2407.16066 |

Features compose independently — both can be enabled simultaneously.

---

## Configuration

All parameters are centralized in `control/config.py` via a `DeePCConfig` dataclass. The noise-adaptive and online-Hankel parameters have safe defaults that are inert when the corresponding features are not activated.

---

## Plant Models

Plant models live in `plants/` and are decoupled from DeePC configuration. Each plant accepts its own physical parameters directly.

| Plant | File | Description |
|-------|------|-------------|
| Kinematic bicycle | `plants/bicycle_model.py` | 2-D vehicle with steering and acceleration inputs. Nonlinear, discrete-time. |

Adding a new plant: create a class in `plants/` with `step(u) -> y`, `output`, and `reset()` methods.

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
- J. Coulson, J. Lygeros, F. Dörfler. *Distributionally Robust Chance Constrained Data-Enabled Predictive Control*. arXiv:1903.06804, 2019.
- J.C. Willems, P. Rapisarda, I. Markovsky, B.L.M. De Moor. *A note on persistency of excitation*. Systems & Control Letters, 2005.
