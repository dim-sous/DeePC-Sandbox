# DeePC Sandbox

A personal research sandbox for **Data-Enabled Predictive Control (DeePC)** on a simulated plant typed
as equations, by default a coupled two-mass spring-damper. An active personal project, not a library or
product. A kinematic bicycle was deleted on 2026-11-11 (no source for its model was known), and the
coupled masses' hand-written plant class on 2026-11-17, replaced by the same system typed as rows.

## About

Model predictive control needs an explicit model of the system. DeePC replaces it with one offline dataset:
excite the system with rich inputs, record what happens, and use block-Hankel matrices of the recorded
trajectories as the model in the quadratic program solved at each step. The foundation the project names is
Willems' fundamental lemma, in the old README's words: "for a linear time-invariant system, a sufficiently
rich input-output trajectory contains all information needed to predict future behavior". How the papers
state it, and where the code departs from them, is in [METHODS.md](METHODS.md), §3.

- **Systems, typed.** The `system` piece defines a system as rows: one per state equation
  ($\dot x_i = f_i(x, u, d, \theta)$), one per output equation ($y_j = h_j(x, \theta)$), and the
  values of the parameter vector $\theta$, in `x1…`, `u1…`, `d1…`, `theta1…`. SymPy reads and checks
  them ([METHODS.md](METHODS.md) §10). A run also analyses the system at a point you give,
  $(x^*, u^*)$, for one input–output pair: its exact linearisation, the poles, the Bode plot and the
  step, impulse and ramp responses (python-control).
- **Open loop.** The `open_loop` piece simulates a saved system run from $x_0$: an i.i.d. uniform
  excitation per input, clipped to the actuator bounds, and a Gaussian disturbance where the
  system's rows name one ($d_1, \dots$), both held over $T_s$ and integrated by RK45; the outputs
  measured as $y_k = h(x_k)$ plus Gaussian noise (§11). Its numbers: the PE order of $u$, each
  output's SNR, the clipped inputs, the step time. Its record is the controller's data.
- **Closed loop.** The `closed_loop` piece runs DeePC on a saved open-loop run (§5). The loop's
  plant is that run's plant (its system, $T_s$, actuator bounds, noise, disturbance and
  tolerances), sampled by the same two functions; the reference is typed as one row in $t$ per
  output ($r_1, \dots$); the loop has its own seed and initial state. The first $T_{ini}$ samples
  apply zero input; a solve with no solution stops the run. Its numbers: the RMSE of
  $h(x_k) - r_k$ per output, the solves by status and by solver, the solve time's mean, spread and
  maximum, $\lVert\sigma_y\rVert_2$, the clipped inputs.
- **Controller.** The DeePC quadratic program in sparse form (inputs and outputs as decision
  variables), with regularisation of the Hankel weights and a slack on the past outputs, and hard
  input bounds (§4). Its code also holds input-rate bounds, soft output bounds, noise-adaptive
  regularisation and an online Hankel window, which no piece reaches since 2026-11-17 (§6, §7).
- **Stack.** CVXPY with OSQP (SCS as the fallback), SymPy, python-control, Benchlab for the records,
  the charts (a still and the Lab's live twin), the `deepc` command and the Lab, NumPy and SciPy.

The equations, the controller, its sources and where the code departs from them are in
[METHODS.md](METHODS.md); the rewrite's plan and the developer's decisions in
[docs/project/design.md](docs/project/design.md). The project is being rebuilt, first with the
scaffolding that will expose its gaps (the rules in [CLAUDE.md](CLAUDE.md), the three-folder layout,
records and a Lab), then the fixes.

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
uv run deepc run system                        # the two masses typed, read and checked -> results/system/default.json
uv run deepc run system --set 'f=["x2", "-sin(x1) - theta1*x2 + u1"]' --set 'h=["x1"]' --set 'theta=[0.2]' --name pendulum
uv run deepc run open_loop                     # that system simulated open loop -> results/open_loop/default.json
uv run deepc run closed_loop                   # DeePC on that run -> results/closed_loop/default.json
uv run deepc run closed_loop --set duration=10 --sweep lambda_g=1,10,100 --name lg
uv run deepc draw results/closed_loop/default.json                       # its charts as PNGs beside it
uv run deepc draw results/closed_loop/lg-*.json --number y1_rmse --vs lambda_g
uv run lab --no-open                           # the Lab on http://127.0.0.1:8765: runs ticked and compared on the left, live charts on the right
```

The Lab runs the pieces for you, from a form over the settings, each run its own process, side by
side up to the machine's cores, and computes nothing itself. Benchlab's own suite drives the page; here one test
checks that the pieces plug into it (`tests/test_bench.py`).

## Checks

```
uv run pytest
```

The suite runs locally and is the gate for every change. It holds:

- **The pieces** (`tests/test_system.py`, `tests/test_open_loop.py`, `tests/test_closed_loop.py`).
  Each against known answers: the parser and the reference reader, what they read and refuse; the
  integrator and both loops' plant against the exact sampled linear system; the loop's plant is
  the open-loop run's; the fundamental lemma on the Hankel blocks of noise-free data; what each
  piece refuses; that its numbers are reductions of its arrays; that a saved run which no longer
  runs again to its file is refused.
- **The solves** (`tests/test_solve_outcomes.py`). Every status counted by its own name, the SCS
  fallback named, no solution flagged.
- **The layout's direction** (`tests/test_roots.py`, `tests/test_devkit_layers.py`). The controller
  (`src/`) imports neither the simulated world (`tools/`) nor the bench (`devkit/`), the world
  never imports the bench, and `src/` holds exactly the modules its inventory declares.
- **Benchlab's check** (`tests/test_bench.py`) on a small run of every piece.
- **The document checks.** CLAUDE.md's length budget and METHODS.md's citations. Every folder path and
  every `file:line` pointer in this README, METHODS.md and failures.md must exist, and each pointer must fall
  inside its file. Whether a pointer still points at the code it describes is not checked.

There is no safety net of frozen numbers between 2026-11-17 and the branch that makes the new one:
the old one froze the old pipeline's numbers and went with it.

## Project layout

| Path | Role |
|---|---|
| `src/deepc/` | the controller, what a real system would run: settings (`config.py`), the constraints it reads (`constraints.py`), the QP (`controller.py`), Hankel matrices (`hankel.py`, `online_hankel.py`), the excitation check (`regularization.py`), the noise estimator (`noise_estimator.py`) |
| `tools/system/` | the simulated world: the parser (`parse.py`, rows to $f$, $h$ and the sizes, and a reference typed in $t$), the one integrator (`integrate.py`, RK45 with $u$ held), the plant's sample (`step.py`: the output measured, then the step with the actuator clip), the linearisation (`linearize.py`) |
| `devkit/` | the bench ([devkit/README.md](devkit/README.md)): the pieces (`devkit/harness/`), declared to Benchlab, which holds the settings and records, the charts, the `deepc` command and the Lab |
| `tests/` | the pieces (`test_system.py`, `test_open_loop.py`, `test_closed_loop.py`), the solves (`test_solve_outcomes.py`), the layout's direction (`test_roots.py`, `test_devkit_layers.py`), that they plug into Benchlab (`test_bench.py`) and the document checks (`test_conventions.py`) |
| `METHODS.md` | what the code computes, as equations, with references |
| `CLAUDE.md` | the working rules |
| `docs/project/failures.md` | the incidents the rules were written after (D1–D18) |
| `references/` | the papers, on each computer only (git-ignored: the repository is public) |

## Limitations

Simulation only. The known problems are recorded in
[docs/project/failures.md](docs/project/failures.md):

- **D1** (fixed, PR #8) `--Ts`, and the tuner's search over it, never reached the plant.
- **D2** (fixed, PR #9) an inaccurate solve was counted as optimal.
- **D3** (fixed, PR #9) solver failures were silent: a hidden SCS retry, a zero input applied, warnings that
  changed nothing, a tuner that scored a failed trial 1e6.
- **D4, D9, D10, D12, D15** (closed 2026-11-17, with the old pipeline) the old loop's outputs had no
  noise, its Hankel columns joined episodes, it aligned the reference differently from the data,
  its data recorded inputs before the clip, and it tracked errors with a zero reference. The new
  loop measures as the data does, on one open-loop run, tracking a reference with $y$.
- **D5, D11, D16, D17** the code of the noise-adaptive and online-Hankel variants, kept in `src/`
  but reached by no piece since 2026-11-17.
- **D6** comments that describe other code.
- **D7** (closed, the bicycle deleted) the bicycle's output bounds sat on the speed error.
- **D8** code nothing calls: the variants' code above is kept on purpose until the maths phase.
- **D13** the excitation check uses a lower order than the fundamental lemma asks for.
- **D14** the default regulariser on the Hankel weights is the squared 2-norm, where Coulson et al. use the
  1-norm and a low-rank step the code lacks.
- **D18** with no rate bound, as the closed loop builds the controller, the SCS fallback found no
  solution where OSQP found one.

**Claims from before the rebuild, not re-measured.** These were written into the old CLAUDE.md before any
run was recorded. No record backs them, and each stays unverified until one re-measures it.

- Absolute or monotonic outputs destroy Hankel conditioning; error-based or shift-invariant
  outputs are used instead.
- Open-loop data collection produces drifted, asymmetric output distributions; a stabilising
  baseline during collection keeps the errors centred.
- Tuning on short simulations and evaluating on long ones exposes drift.
- Sampling time (Ts) dominates tracking accuracy more than any regularisation parameter. The
  runs behind it went through `--Ts` before it reached the plant (D1, fixed in PR #8).
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
