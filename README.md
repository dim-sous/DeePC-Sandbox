# DeePC Sandbox

A personal research sandbox for **Data-Enabled Predictive Control (DeePC)** on two simulated plants, a
kinematic bicycle and a coupled two-mass spring-damper. An active personal project, not a library or product.

## About

Model predictive control needs an explicit model of the system. DeePC replaces it with one offline dataset:
excite the system with rich inputs, record what happens, and use block-Hankel matrices of the recorded
trajectories as the model in the quadratic program solved at each step. The foundation the project names is
Willems' fundamental lemma, in the old README's words: "for a linear time-invariant system, a sufficiently
rich input-output trajectory contains all information needed to predict future behavior". How the papers
state it, and where the code departs from them, is in [METHODS.md](METHODS.md), §3.

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
(the rules in [CLAUDE.md](CLAUDE.md), a safety net, the three-folder layout, then records and a Lab), then
the fixes.

## Installation

Needs [uv](https://docs.astral.sh/uv/). From the project folder:

```
uv sync
```

This creates `.venv/` with Python 3.13 (pinned in `.python-version`) and the library versions pinned in
`uv.lock`, and installs the three code folders (`src/`, `tools/`, `devkit/`) so their packages import by
name. After pulling a change that moved packages, run `uv sync` once.

## Usage

Each run of a piece saves one record, a JSON file with the settings it used, everything it produced,
its numbers and the commit; [devkit/README.md](devkit/README.md) has an example of every command:

```bash
uv run deepc run dataset                       # training data from the bicycle -> results/dataset/default.json
uv run deepc run closed_loop                   # DeePC on that data -> results/closed_loop/default.json
uv run deepc run closed_loop --set sim_duration=10 --sweep lambda_g=1,10,100 --name lg
```

The older command still writes the HTML report:

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

**Adding a plant.** Implement `PlantBase` from `tools/plants/base.py`:

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

Then add it to `PLANT_REGISTRY` in `tools/plants/__init__.py` and run with `--plant my_plant`.

## Checks

```
uv run pytest
```

The suite runs locally and is the gate for every change. It holds:

- **The safety net** (`tests/reference/`). Today's numbers are frozen: training data, reference paths,
  five closed loops and `run.py` end to end. The test compares the code against them byte for byte, along
  with every setting each case ran under.
- **The layout's direction** (`tests/test_roots.py`). The controller (`src/`) imports neither the plants
  (`tools/`) nor the bench (`devkit/`), the plants never import the bench, and `src/` holds exactly the
  modules its inventory declares.
- **The pieces** (`tests/test_harness.py`). Each reproduces the frozen numbers bit for bit through its
  saved file, its numbers are reductions of its arrays, and a saved dataset run that no longer runs again
  to its file is refused.
- **The document checks.** CLAUDE.md's length budget and METHODS.md's citations. Every folder path and
  every `file:line` pointer in this README, METHODS.md and failures.md must exist, and each pointer must fall
  inside its file. Whether a pointer still points at the code it describes is not checked.

The frozen files are remade only by `uv run python -m tests.reference.make`, on a change meant to move the
numbers.

## Project layout

| Path | Role |
|---|---|
| `src/deepc/` | the controller, what a real system would run: settings (`config.py`), the constraints it reads (`constraints.py`), the QP (`controller.py`), Hankel matrices (`hankel.py`, `online_hankel.py`), the excitation check (`regularization.py`), the noise estimator (`noise_estimator.py`) |
| `tools/plants/` | the simulated world: `PlantBase` (`base.py`), the bicycle (`bicycle_model.py`), the coupled masses (`coupled_masses.py`), the excitation signals (`excitation.py`), the plant registry (`__init__.py`) |
| `devkit/` | the bench ([devkit/README.md](devkit/README.md)): the simulation code (`devkit/sim/`), settings and records (`devkit/record/`), the pieces `dataset` and `closed_loop` (`devkit/harness/`), the `deepc` command (`devkit/terminal/`) |
| `run.py` | the command: data collection, closed loop, metrics, HTML report |
| `tune.py` | Bayesian search over the controller's settings (Optuna) |
| `tests/` | the safety net (`tests/reference/`, `test_reference.py`), the layout's direction (`test_roots.py`, `test_devkit_layers.py`), the records and pieces (`test_record.py`, `test_harness.py`) and the document checks (`test_conventions.py`) |
| `METHODS.md` | what the code computes, as equations, with references |
| `CLAUDE.md` | the working rules |
| `docs/project/failures.md` | the incidents the rules were written after (D1–D17) |
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
- **D13** the excitation check uses a lower order than the fundamental lemma asks for.
- **D14** the default regulariser on the Hankel weights is the squared 2-norm, where Coulson et al. use the
  1-norm and a low-rank step the code lacks.
- **D15** tracking on errors with a zero reference, where Coulson et al. track a reference with the
  system's outputs.
- **D16** the noise-adaptive rule is not the one in the paper it cites.
- **D17** the online window is not the one in the paper it cites.

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

The main sources, checked against their arXiv versions on 2026-11-11 except the last, which is not on arXiv;
the full list is in [METHODS.md](METHODS.md#references).

- J. Coulson, J. Lygeros, F. Dörfler, "Data-Enabled Predictive Control: In the Shallows of the DeePC",
  arXiv:1811.05890v2, 2019 (no venue on the record).
- J. Coulson, J. Lygeros, F. Dörfler, "Regularized and Distributionally Robust Data-Enabled Predictive
  Control", arXiv:1903.06804v2, 2019.
- H. J. van Waarde, C. De Persis, M. K. Camlibel, P. Tesi, "Willems' Fundamental Lemma for State-space
  Systems and its Extension to Multiple Datasets", arXiv:2002.01023v2, 2020.
- J. C. Willems, P. Rapisarda, I. Markovsky, B. L. De Moor, "A note on persistency of excitation",
  *Systems & Control Letters* 54(4), 325–329, 2005 (not read).

## License

[MIT](LICENSE)
