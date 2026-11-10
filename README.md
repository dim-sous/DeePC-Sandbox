# DeePC Sandbox

A personal research sandbox for **Data-Enabled Predictive Control (DeePC)** on two simulated plants, a
kinematic bicycle and a coupled two-mass spring-damper. An active personal project, not a library or product.

## About

Model predictive control needs an explicit model of the system. DeePC replaces it with one offline dataset:
excite the system with rich inputs, record what happens, and use block-Hankel matrices of the recorded
trajectories as the model in the quadratic program solved at each step. The foundation the project names is
Willems' fundamental lemma, in the old README's words: "for a linear time-invariant system, a sufficiently
rich input-output trajectory contains all information needed to predict future behavior" (not yet checked
against its source; see [METHODS.md](METHODS.md), §3).

- **Plants.**

  | Plant | Type | Inputs | Outputs | Scenarios |
  |-------|------|--------|---------|-----------|
  | `bicycle` | Nonlinear | steering, acceleration | lateral, heading and speed errors | `default` (sinusoid along a line), `lissajous` |
  | `coupled_masses` | LTI MIMO | forces $F_1$, $F_2$ | position and velocity errors of both masses | `default` (steps), `sinusoidal` |

- **Data.** Several episodes per plant, each driven by a proportional baseline controller plus excitation,
  with Gaussian noise on the recorded outputs.
- **Controller.** The DeePC quadratic program in sparse form (inputs and outputs as decision variables), with
  regularisation of the Hankel weights and a slack on the past outputs, hard input and rate bounds, and
  optional soft output bounds. The outputs are tracking errors and the reference is zero.
- **Variants**, both off by default: noise-adaptive regularisation (`--noise-adaptive`) and an online
  Hankel window fed by closed-loop data (`--online-hankel`).
- **Tuning.** A Bayesian search over the controller's settings with Optuna.
- **Stack.** CVXPY with OSQP (SCS as the fallback), Optuna's TPE sampler, Plotly for the reports, NumPy
  and SciPy.

The equations, the data collection, the controller, its variants, the metrics and their sources are in
[METHODS.md](METHODS.md). The project is being rebuilt, first with the scaffolding that will expose its gaps
(the rules in [CLAUDE.md](CLAUDE.md), a safety net, then records and a Lab), then the fixes.

## Installation

Needs [uv](https://docs.astral.sh/uv/). From the project folder:

```
uv sync
```

This creates `.venv/` with Python 3.13 (pinned in `.python-version`) and the library versions pinned in
`uv.lock`.

## Usage

```bash
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
```

Each run writes a self-contained HTML report with interactive Plotly charts and a metrics file to
`results/<tag>/`.

**Adding a plant.** Implement `PlantBase` from `plants/base.py`:

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

## Checks

```
uv run pytest
```

The suite runs locally and is the gate for every change. It holds:

- **The safety net** (`tests/reference/`). Today's numbers are frozen: training data, reference paths,
  five closed loops and `run.py` end to end. The test compares the code against them byte for byte, along
  with every setting each case ran under.
- **The document checks.** CLAUDE.md's length budget and METHODS.md's citations. Every folder path and
  every `file:line` pointer in this README, METHODS.md and failures.md must exist, and each pointer must fall
  inside its file. Whether a pointer still points at the code it describes is not checked.

The frozen files are remade only by `uv run python -m tests.reference.make`, on a change meant to move the
numbers.

## Project layout

| Path | Role |
|---|---|
| `run.py` | the command: data collection, closed loop, metrics, HTML report; the plant registry |
| `tune.py` | Bayesian search over the controller's settings (Optuna) |
| `control/` | the controller: settings (`config.py`), the QP (`controller.py`), Hankel matrices (`hankel.py`, `online_hankel.py`), the excitation check (`regularization.py`), the noise estimator (`noise_estimator.py`) |
| `plants/` | `PlantBase` (`base.py`), the bicycle (`bicycle_model.py`), the coupled masses (`coupled_masses.py`) |
| `sim/` | data collection (`data_generation.py`), scenarios (`scenarios.py`), the closed loop (`simulation.py`) |
| `tests/` | the safety net (`tests/reference/`, `test_reference.py`) and the document checks (`test_conventions.py`) |
| `METHODS.md` | what the code computes, as equations, with references |
| `CLAUDE.md` | the working rules |
| `docs/project/failures.md` | the incidents the rules were written after (D1–D12) |
| `references/` | the papers, on each computer only (git-ignored: the repository is public) |

## Limitations

Simulation only. The known problems are recorded, none fixed yet, in
[docs/project/failures.md](docs/project/failures.md):

- **D1** `--Ts`, and the tuner's search over it, never reaches the plant.
- **D2** an inaccurate solve is counted as optimal.
- **D3** solver failures are silent: a hidden SCS retry, a zero input applied, warnings that change nothing.
- **D4** the training outputs are noisy and the closed-loop outputs are not.
- **D5** the online Hankel gate reads an estimator that only runs under `--noise-adaptive`.
- **D6** comments that describe other code.
- **D7** the bicycle's output bounds sit on the speed error.
- **D8** code nothing calls.
- **D9** Hankel columns that join two episodes, or the training data and the loop.
- **D10** the loop and the data align the reference with the state differently.
- **D11** the noise-adaptive tightening enters no constraint at the defaults.
- **D12** the bicycle's training data records inputs before the plant clips them.

**Claims from before the rebuild, not re-measured.** These were written into the old CLAUDE.md before any
run was recorded. No record backs them, and each stays unverified until one re-measures it.

- Absolute or monotonic outputs destroy Hankel conditioning; error-based or shift-invariant
  outputs are used instead.
- Open-loop data collection produces drifted, asymmetric output distributions; a stabilising
  baseline during collection keeps the errors centred.
- Tuning on short simulations and evaluating on long ones exposes drift.
- Sampling time (Ts) dominates tracking accuracy more than any regularisation parameter. The
  runs behind it went through `--Ts`, which never reaches the plant (D1).
- One-at-a-time parameter sweeps miss interactions; a Bayesian joint search finds better
  configurations.
- Training data quality matters more than algorithm complexity: a well-conditioned Hankel
  matrix with good coverage beats any amount of regularisation tuning.

## References

The main sources, none yet checked against its source here; the full list is in
[METHODS.md](METHODS.md#references).

- J. Coulson, J. Lygeros, F. Dörfler, *Data-Enabled Predictive Control: In the Shallows of the DeePC*,
  2019 (venue not checked).
- J. C. Willems, P. Rapisarda, I. Markovsky, B. L. M. De Moor, *A note on persistency of excitation*,
  *Systems & Control Letters*, 2005.

## License

[MIT](LICENSE)
