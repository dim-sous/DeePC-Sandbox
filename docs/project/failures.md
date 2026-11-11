# The failures the rules were written after

`CLAUDE.md` states the rules. This file keeps the incidents, because **a rule without its
incident degrades into a slogan** — every rule tagged `[D…]` was written after something here
was found in this repository. Rules taken from the sibling repositories cite theirs (BESS
`[B…]`, RCWS `[A…]`).

Referenced from `CLAUDE.md` as `[D1]`–`[D17]`. The labels are stable: add new ones, do not
renumber. Each entry states what was observed, with the lines; what it means is the rule's job.

D1–D8 were found on 2026-11-10, reading the code at `45dc91b` (untouched since 2026-05-29) and
running its own functions; their line numbers are that commit's. D9–D12 were found the same day
while writing `METHODS.md` from the code at `3067d03` (the same code), D12 by its code review.
D13–D17 were found on 2026-11-11, reading the papers `METHODS.md` cites (its References, arXiv
versions; numbers in brackets refer to that list). **None is fixed**: the safety net
(`tests/reference/`) freezes each as it behaves, and each waits for the Lab to show it before a
branch fixes it and moves the frozen numbers on purpose.

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

## D9 — Hankel columns that join two runs

- The training data is collected in episodes, each starting from its own reset state
  (`sim/data_generation.py:88`), and the episodes are stacked end to end into one record
  (`:114`). The Hankel matrix is built over the stacked record
  (`control/online_hankel.py:49-50`, through `control/hankel.py:6-37`), so a column whose window
  spans an episode boundary holds samples from two episodes.
- Counted at the plants' defaults (5 episodes; bicycle `T_data` 400, $L$ = 18; coupled masses
  `T_data` 600, $L$ = 24): 68 of the bicycle's 383 columns and 92 of the masses' 577 span a
  boundary.
- The same holds for the online window (`--online-hankel`): its buffer is seeded with the last
  $L - 1$ training samples (`control/online_hankel.py:79-82`) and then receives the loop's
  samples from step `hankel_warmup_steps` on (default 100), so its first $L - 1$ online columns
  join the end of the training data to the loop.

- The literature's construction for several datasets, found 2026-11-11: the mosaic-Hankel matrix,
  one Hankel block per dataset side by side ([6, Definition 2, eq. (8)], with the fundamental
  lemma extended to it in [6, Theorem 2]; [3, Definition 3, eq. (4)] uses the same). Measured on
  the frozen training data (`tests/reference/*-data.npz`; seed 42, plant defaults), the
  per-episode mosaic of the inputs has full row rank at order $L$: 48 of 48 rows over 485
  columns for the coupled masses, 36 of 36 over 315 for the bicycle.

Rules: literature-grounded — several datasets are combined the way the literature does it
(`METHODS.md` §3); a data matrix states what each of its columns holds.

## D10 — the loop and the data align the reference differently

- Closed loop: after step $i$ the state, at time $(i+1)T_s$, is compared with waypoint $r_i$,
  computed at time $iT_s$ (`sim/simulation.py:78-85`, `:139-141`; the scenarios'
  $t_k = kT_s$, `plants/bicycle_model.py:285-327`).
- Data collection (bicycle): after step $k$ the state is compared with the nominal waypoint at
  $(k+1)\,v_{ref}T_s$ (`plants/bicycle_model.py:371-374`, `sim/data_generation.py:97-105`), the
  same instant.
- The bicycle's position RMSE compares positions at equal times
  (`sim/simulation.py:76`, `:193`; `plants/bicycle_model.py:274-275`).
- Not measured.

Rules: one quantity, one formula — the output the data holds and the output the loop gives the
controller come from one function, or the divergence is declared where it lives.

## D11 — a tightening that enters no constraint

- Under `--noise-adaptive` the loop computes an output-bound tightening from the noise estimate
  and passes it to the controller every step (`sim/simulation.py:118-121`).
- The tightening enters only the soft output bounds, which the QP holds only when
  `lambda_out > 0` (`control/controller.py:182-199`), and only on outputs with a finite bound
  (`:187-188`). Every plant's default is 0, so at the defaults it enters no constraint; only the
  scaling of `lambda_g` and `lambda_y` acts. The coupled masses have no output bounds
  (`plants/coupled_masses.py:135-136`), so for them it enters no constraint at any `lambda_out`.

Rules: a setting the run did not use is refused or recorded as unused, never accepted.

## D12 — training inputs the plant never applied

- During data collection the input $u_k$ = baseline + excitation is passed to the plant
  (`sim/data_generation.py:101-102`) and recorded as it was passed (`:108`). The bicycle clips
  its inputs inside its step (`plants/bicycle_model.py:178-179`) to $|\delta| \le 0.5$ rad and
  $a \in [-1, 1]$ m/s²; its excitation alone reaches ±0.5 rad and ±2.5 m/s²
  (`plants/bicycle_model.py:335-358`).
- In the frozen bicycle training data (`tests/reference/bicycle-data.npz`; seed 42, plant
  defaults, `T_data` 400) 12.0 % of the steering samples and 33.8 % of the acceleration samples
  lie outside the clip. In the coupled masses' (`tests/reference/coupled_masses-data.npz`) none
  do.
- The Hankel input blocks are built from the recorded inputs (`control/online_hankel.py:49`).

Rules: one quantity, one formula — the input the data records is the input the plant applied,
or the divergence is declared where it lives.

## D13 — persistency of excitation checked at a lower order than the lemma asks

- The code checks $\mathrm{rank}\, H_L(u) \ge Lm$, order $L = T_{ini} + N$
  (`control/regularization.py:32-33`, called at `control/controller.py:75`).
- [1, §V-A] assumes the input "persistently exciting of order $T_{ini} + N + n(\mathcal B)$";
  [1, Lemma 4.2] and [6, Theorem 1] state the lemma under order $t + n(\mathcal B)$ and $n + L$.
- Measured on the frozen training data (`tests/reference/*-data.npz`; seed 42, plant defaults),
  the stacked inputs have full row rank at both orders: coupled masses 48 of 48 at $L$ = 24 and
  56 of 56 at $L + 4$ = 28; bicycle 36 of 36 at $L$ = 18 and 44 of 44 at $L + 4$ = 22 (4 is the
  state dimension; the lemma is stated for linear systems, which the bicycle is not).

Rules: literature-grounded — a condition the method rests on is checked as the source states it.

## D14 — the regulariser on $g$, and no low-rank step

- Both plants default to `reg_norm_g = "L2"` (`plants/bicycle_model.py:220-234`,
  `plants/coupled_masses.py:139-153`), which the QP implements as the squared 2-norm,
  `cp.sum_squares(self.g)` (`control/controller.py:139-142`).
- [1, eq. (8)] regularises with $\lambda_g \lVert g \rVert_1$; [2, Remark 4.3] pairs robustness in
  the 2-norm sense with regularising by the 2-norm itself.
- [1, eq. (8)] also replaces the data matrix by "a low-rank matrix approximation" of it; the code
  builds its blocks from the raw Hankel matrix (`control/online_hankel.py:49-50`, `:86-90`).

Rules: literature-grounded — a departure from the cited method is declared where it lives
(`# MISMATCH`).

## D15 — tracking on errors with a zero reference

- [1, eq. (6)] penalises $\lVert y_k - r_{t+k} \rVert_Q^2$, with $y$ the system's outputs and $r$
  the reference.
- The code's outputs are errors against a reference (`plants/bicycle_model.py:46-51`,
  `plants/coupled_masses.py:127`) and its cost reference is zero (`sim/simulation.py:126`); the
  training errors are taken against a nominal reference (bicycle: a straight line at $v_{ref}$,
  `plants/bicycle_model.py:371-374`; coupled masses: zero), the closed-loop errors against the
  scenario's (`sim/simulation.py:141`).
- The old CLAUDE.md recorded this as a lesson ("use error-based or shift-invariant outputs"),
  backed by no record (README, "Claims from before the rebuild").

Rules: literature-grounded — a departure from the cited method is declared where it lives and
re-measured against the method it departs from.

## D16 — a noise-adaptive rule the cited paper does not state

- `control/noise_estimator.py:1-5` cites arXiv:1903.06804 ([2]) for scaling "the Wasserstein
  robustness radius (lambda_g, lambda_y)" from estimated noise. The code multiplies
  `lambda_g` and `lambda_y` at every step by $\mathrm{clip}(\mathrm{mean}(\hat\sigma)/\sigma_0, 1, 10)$,
  $\hat\sigma$ from a window of its own prediction residuals, and tightens the output bounds
  (`sim/simulation.py:107-121`, `control/noise_estimator.py:36-46`).
- [2] relates $\lambda_g$ to the radius $\epsilon$ (after [2, Theorem 4.2]), says to choose
  $\epsilon$ "in a data-driven fashion" ([2, Remark 4.1]), and in [2, §V] fixes $\epsilon$ for a
  run and compares runs over 15 random simulations per $\epsilon$. Its text, searched for
  "estimat", "tighten" and "adaptive" (`pdftotext` of arXiv v2) and read around each hit, mentions
  neither an online noise estimate nor an output tightening.

Rules: literature-grounded — a citation names the source of what the code does, and a rule of
our own is said to be ours.

## D17 — an online window cited to a paper that describes another

- `control/online_hankel.py:9` cites arXiv:2407.16066 ([3]). The code slides a fixed-width
  window, dropping the oldest column for each new one (`control/online_hankel.py:109-114`),
  and gates updates on the noise estimate (D5).
- [3, Algorithm 1] grows a mosaic-Hankel matrix and keeps a new column only if the minimum
  non-zero singular value stays at or above a threshold; [3, §III-B] reduces its order by a
  numerical SVD. [3, §I] attributes the sliding window to Baros et al. (*Automatica* 138, 109926,
  2022, its reference 18; not read here) and says that method "requires a PE control input in
  real-time".

Rules: literature-grounded — a citation names the source of what the code does.

