# The failures the rules were written after

`CLAUDE.md` states the rules. This file keeps the incidents, because **a rule without its
incident degrades into a slogan** — every rule tagged `[D…]` was written after something here
was found in this repository. Rules taken from the sibling repositories cite theirs (BESS
`[B…]`, RCWS `[A…]`).

Referenced from `CLAUDE.md` as `[D1]`–`[D8]`. The labels are stable: add new ones, do not
renumber. Each entry states what was observed, with the lines; what it means is the rule's job.

D1–D8 were found on 2026-11-10, reading the code at `45dc91b` (untouched since 2026-05-29) and
running its own functions; their line numbers are that commit's. **None is fixed**: the safety net (`tests/reference/`) freezes each
as it behaves, and each waits for the Lab to show it before a branch fixes it and moves the
frozen numbers on purpose.

---

## D1 — `--Ts` never reaches the plant

- `run.py:565` and `tune.py:332` build the plant with no arguments, so it integrates at its
  constructor's default step: bicycle `Ts = 0.1` (`plants/bicycle_model.py:95`, used at `:182`),
  coupled masses `Ts = 0.05` (`plants/coupled_masses.py:38`, discretised at `:73`).
- `build_deepc_config` sets the config's `Ts` from the plant, then lets the override replace it
  (`control/config.py:92-93`). What reads the config's `Ts`: the step count
  (`control/config.py:68`), the reference's spacing (`sim/scenarios.py:31`), the results' time
  axis (`sim/simulation.py:179`) and the report's plots (`run.py:120`, `:593`,
  `plants/bicycle_model.py:480`). What reads the plant's: its own step, the excitation signals
  (`plants/bicycle_model.py:352`, `plants/coupled_masses.py:268`) and the data-collection
  reference (`sim/data_generation.py:97`).
- `tune.py:63` searches `Ts` over [0.02, 0.2] through this path; the README's example runs
  `--Ts 0.039` (README, Quick Start); the old CLAUDE.md held the lesson "Sampling time (Ts)
  dominates tracking accuracy more than any regularization parameter". No record of the runs
  behind the lesson exists.

Measured with the repository's functions as `run.py` calls them — bicycle, scenario `default`,
seed 42, `sim_duration` 10 s (default 60 s), every other setting at the plant's default:

| `--Ts` | config `Ts` | plant `Ts` | steps | plant travel in x | reference travel in x | `rmse_position` |
|---|---|---|---|---|---|---|
| (default) | 0.1 | 0.1 | 100 | 64.2 m | 50.7 m | 6.84 m |
| 0.039 | 0.039 | 0.1 | 256 | 101.5 m | 50.2 m | 39.3 m |

Coupled masses, same conditions (`sim_duration` 10 s, default 30 s), `--Ts 0.1`: config 0.1,
plant 0.05, 100 steps.

Rules: a setting the run did not use is refused or recorded as unused, never accepted silently;
name the conditions every number came from; the plant is never tuned to make the controller
look right.

## D2 — an inaccurate solve counted as optimal

- `run.py:92` and `sim/simulation.py:195` count a status as optimal when it contains the word
  `"optimal"`, so OSQP's `optimal_inaccurate` counts.
- The frozen `bicycle-default` loop (`tests/reference/bicycle-default.npz`, 5 s) holds 3
  `optimal_inaccurate` of 50 statuses; `bicycle-noise-adaptive` 4 of 50,
  `bicycle-online-hankel` 3 of 50. `compute_metrics` reports each as 100 % optimal.
- Measured at 10 s (conditions as D1): bicycle 3 of 100, coupled masses 1 of 200, both
  reported 100 %.

Rules: fail loud — a solver status is counted by its own name in the record, never folded
into another.

## D3 — failures that make no sound

- `control/controller.py:260` catches every exception from the OSQP solve, `Exception`
  included, and solves again with SCS (`:261-262`); nothing records that it happened.
- `control/controller.py:280-281`: when the solve leaves no solution, the controller returns
  an input of zeros and the loop applies it to the plant.
- The persistent-excitation check prints a warning and the controller is built anyway
  (`control/regularization.py:36`, `control/controller.py:75-77`).
- `tune.py:96-98` turns any exception in a trial into the score 1e6; `tune.py:87` does the
  same for a NaN or infinite score.

Rules: fail loud — no swallowed exception, no stand-in answer for one the code failed to
compute; a fallback is counted in the record; every guard makes a failure less legible.

## D4 — one measurement, two formulas

- Training data: the recorded output is the plant's error plus Gaussian noise of
  `noise_std_output` (`sim/data_generation.py:105-106`).
- The baseline controller that runs during data collection reads the error before any noise
  is added (`sim/data_generation.py:98-100`).
- Closed loop: the output DeePC receives is the error with no noise
  (`sim/simulation.py:81`, `:141`).

Rules: one quantity, one formula, or each divergence declared where it lives (`# MISMATCH`);
the controller receives only what the plant reports.

## D5 — the online Hankel gate reads a switch it does not name

- Under `--online-hankel`, a column is added only while the mean of the noise estimator's
  standard deviations is below `1.0` (`sim/simulation.py:144-147`); no source states the
  threshold.
- The estimator is updated only under `--noise-adaptive` (`sim/simulation.py:107-109`).
  Without it, its standard deviations stay at their initial zeros and the gate passes at
  every step.

Rules: a change in A must not require a change in B; a guard states its source and defaults
to reporting, not acting.

## D6 — comments that describe other code

- `control/online_hankel.py:53`: "max_cols: 0 means use offline size + sim_steps headroom";
  the lines below use the offline size alone.
- `plants/bicycle_model.py:30`: the reference point is the "nearest waypoint"; the loop passes
  the waypoint at the current step's index (`sim/simulation.py:141`, `path[step_idx]`).

Rules: a comment states what the code does, checked against the lines below it, the day it is
written.

## D7 — the bicycle's output bounds sit on the speed error

- `plants/bicycle_model.py:215-217` bounds `e_velocity` (speed minus reference speed) to
  [0, 10], so with the bounds active the controller is asked to keep the speed at or above the
  reference speed.
- They are active only while `lambda_out > 0` (`control/controller.py`, the output-constraint
  block); every plant's default is 0.

Whether bounds on the speed itself were meant is the developer's question; the code does not
say.

Rules: a bound names the quantity it bounds, with its unit.

## D8 — code nothing calls

`DeePCConfig.Q` and `.R` (`control/config.py:75`, `:79`), `build_data_matrices`
(`control/hankel.py:40`; `control/online_hankel.py` splits the blocks itself),
`SlidingHankelWindow.get_pe_rank` and `.get_sigma_min` (`control/online_hankel.py:145`,
`:148`), `BicycleModel.position`, `.heading`, `.speed` (`plants/bicycle_model.py:162-171`), and
the locals `rmse_x1`, `rmse_x2` (`plants/coupled_masses.py:200-201`). Found by searching the
repository for each name's callers.

Rules: no code stays that the developer cannot explain; one variant, one file, and the
registry is the only place that knows which exist.
