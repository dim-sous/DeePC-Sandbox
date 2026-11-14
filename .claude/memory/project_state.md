---
name: project-state
description: Where the DeePC-Sandbox rebuild stands — what is merged, what comes next, what is still open for the developer
metadata:
  type: project
  modified: 2026-11-14
---

The developer came back to this repository on 2026-11-10 after months away, remembering neither
the maths nor the code, only that the project "was not working as intended". The rebuild follows
BESS's and RCWS's: scaffolding first (rules, safety net, records, a Lab) to expose the gaps, then
the fixes. PRs #1–#12 ran in one cloud session; from 2026-11-11 evening the work runs in VS Code
on the developer's computer ([[multi-device]]).

**Merged** (`git log --oneline --first-parent`):

| PR | What |
|---|---|
| #1 | Rules (CLAUDE.md, `failures.md` D1–D8) and the safety net (`tests/reference/`) |
| #2 | Documentation, Furuta-style (README, METHODS.md from the code) |
| #3 | METHODS.md checked against the five papers (D9–D17; `references/` is git-ignored) |
| #4 | Layout move: `src/` ← `tools/` ← `devkit/` |
| #5 | Records: settings, saved runs, `deepc run`, pieces `dataset` and `closed_loop` |
| #6 | Charts: the figure layer and `deepc draw` |
| #7 | The Lab (BESS's, ported) |
| #8 | D1 fixed: T_s is the dataset run's step, for its data and every loop on it |
| #9 | D2 + D3 fixed by reporting: solves counted by status, solver and zero input; `tune.py` fails loud |
| #10 | `run.py` and the plants' plotting retired; Path chart and input-bound lines in the Lab |
| #11 | The smaller Lab (the developer's cut, [[project-scope]]) |
| #12 | Session notes in the repo (`.claude/memory/`) |
| #13 | The bicycle deleted (no source for its model); position history, Position RMSE, Path chart gone; safety net gains the masses' online-Hankel and noise-adaptive cases, remade on the developer's computer |
| #14 | Rewrite branch 1: the `system` piece (typed rows x', y, θ; Lab row editor), `tools/system/` (SymPy parser, RK45 `hold`), the no-units rule, `docs/project/design.md` |
| #15 | Rewrite branch 2: the `open_loop` piece (excitation, actuator clip, typed disturbances d, y_k = h(x_k) + noise; numbers PE order, SNR, clipped, step time); the old `dataset` kept until branch 3 |
| #16 | System analysis S1: the Analysis group (x*, u*, pair), exact linearisation, poles, Bode (declared log axes), step/impulse/ramp, DC gain, margins; python-control 0.10.2 |

**Moved onto Benchlab (2026-11-14, PR #17):** the developer's shared Lab for DeePC,
Hydrofoil and BESS (`github.com/dim-sous/benchlab`; its `docs/design.md` and
`docs/plan-branch-1.md`). Its branch 1 moves `devkit/record`, `figure`, `terminal` and `lab` there
unchanged but for the four ties to DeePC (the `deepc` name, the title, the `harness` import, the
provenance's repo and libraries); DeePC installs Benchlab and declares itself in
`devkit/harness/__init__.py` and `pyproject.toml`'s entry point. Built 2026-11-14; every record
and still checked identical to `master`'s but for provenance and the wall-clock fields two runs of
`master` already differ in. Merged the same day on the developer's word: dim-sous/BenchLab#1 merged and
tagged `v0.1.0`, the pin moved to the tag, the whole suite run again, then #17 merged. To work on
Benchlab and DeePC together on one computer, point `[tool.uv.sources]` at the local copy for the
while, never committed (Benchlab's `README.md`). From then on a change to the page, the charts, the records or the command is
made in Benchlab and taken here by raising its pin.

**Benchlab branch 2a, runs in progress (2026-11-14, PR open here and as dim-sous/BenchLab#2, Benchlab
pinned to its branch `runs-in-progress` until it is tagged v0.2.0):** the Lab runs each run as
its own process (`python -m benchlab.runner`), up to the machine's cores side by side, the rest
queued, shown in a strip along the bottom in place of the terminal (Queued, Running with its time,
Done, Failed with its words, Stopped; Stop and ✕). DeePC's side: the Lab's tests that drove the
old terminal moved to Benchlab or went with it; the browser test now reads the settings from the
record a run saved, and its chart test waits until the charts are drawn (read at a fixed 1.5 s,
they were none twice in the whole suite); `docs/project/lab-words.md` holds the strip's words. Measured: a λ_g sweep of
three closed loops at `sim_duration=10`, 43.7 s one after another before, about 32 s side by side.
Handed off here and approved by the developer. On Benchlab installed from its pushed branch: every
piece's records and stills as before but for provenance and wall-clock fields; the whole suite
244 passed (268 on `master`: 26 tests of the old terminal left, 2 checks added).
Benchlab's branches 2b (words as slots, no hovers), 2c (the tree) and 2d (the cards) follow, each
driven here before the next.

**Now (2026-11-11 evening): the target design.** The developer: "dont fixate on the current state of
the codebase at all ... The only sure thing is the aforementioned scope: define plant -> simulate
plant with chosen configurations -> (attempt to) control the plant with deepc under different
configuration. Literally everything else can be thrown out." Drafted in `docs/project/design.md`
on branch `system-generator` (three pieces system → dataset → closed_loop, typed ODEs, RK45, DeePC
as [1, eq. (8)] tracking y, names u/y/x, no units), every choice decided by the developer the same
evening. The rewrite is three branches: (1) `system-generator` — the `system` piece
(`devkit/harness/system/`) and `tools/system/` (parser, RK45 `hold`), SymPy added, the units rule
changed in CLAUDE.md and the figure skill. Handed off once with f/h as `;` lines, x0, bounds and a
free-response chart; the developer redid it ("the current way to define the odes feels off"):
labelled rows (+/✕, a new Lab row editor for list settings declaring `rows`), θ values in the
System tab, no chart, no x0/bounds — rebuilt, merged as PR #14 on 2026-11-11; (2) `open_loop`
(branch `open-loop`, built and handed off 2026-11-11): a saved system run (`harness.system.
system_of`, checked) simulated from x0, u_k i.i.d. uniform on its own range per input (as [1,
§VI-B]), clipped to the actuator bounds and counted, y_k = h(x_k) + noise (as [1, eq. (1)]; the
draft's h(x_{k+1}) was the old code's), true x recorded; the old `dataset` stays until (3) — the
developer's choices 2d–2g in design.md. After the first hand-off they asked for process noise
(typed disturbances d1… in System rows, σ_d in Open loop, drawn per sample and held: 2h) and
useful numbers (PE order of u by the controller's own check, SNR per output, clipped, step time:
2i) — merged as PR #15 on 2026-11-11; (3) the closed loop on [1, eq. (8)] tracking y, deleting the old pipeline and
making a new safety net. The queue below predates it and is superseded where the design says so.

**System analysis, in three branches (the developer, 2026-11-11: "lets take it in order: system
-> open loop -> closed loop"):** S1 on branch `system-analysis` — the System tab's Analysis group
(x*, u* default the origin, the pair u_j -> y_i), the exact SymPy linearisation, python-control
0.10.2 (added) for poles, Bode (log axes now declarable on a Series), step/impulse/ramp, DC gain,
margins; numbers per their pick; merged as PR #16 on 2026-11-11 (the developer: "Once you merge
we stop there, enough for today"). **Pick up here:** S2 Nyquist, Nichols, root locus (charts of
one quantity against another, root-locus branches each on its own x: the figure layer needs
per-trace abscissae again); S3 equilibria and their type, phase portraits (a vector field). Then
more Open loop if asked, then the closed loop.

**Asked for next (2026-11-11, during the open-loop review):** plots on the System tab "useful
for understanding the system (bode plots, Nyquist plots, Nichols plots, Root locus plots,
step/impulse/ramp response plots, 2d phase portraits etc)". Its own branch after `open-loop`.
Decided the same evening: "the linearization is not fixed but defined by the user during plot
request" — the operating point (x*, u*) is part of each plot request, not of the system run.
And: "the system tab number should be number that reflect the system (critical points and their
nature, etc)" — equilibria f(x*, u*, 0) = 0 and their type from the Jacobian's eigenvalues, in
place of the sizes n, m, p, q, r. Open: which input-output pairs for a MIMO system, the library
(python-control, a mature one, would be a new dependency), how a plot request is made in the Lab,
at which u and in what region equilibria are sought (a pendulum has one at every kπ), symbolic
(SymPy) or numerical solving.

**Next, in the order agreed** (the developer picks; propose, then plan mode, then build):
1. **Test speed** — the suite took 148 s on 2026-11-11, measured with `--durations`: about 70 %
   simulating closed loops (the same loops rebuilt in several test files), 22 % the browser
   tests. Proposed: pytest-xdist `-n auto` as RCWS's `pyproject.toml` does (4 cores in the cloud
   container), the tests sharing a port or the Lab's queue kept on one worker; and
   the shared runs built once per suite. No number moves.
2. **Fewer charts and numbers per record** — proposed, not yet confirmed in detail: charts =
   each output, each input (with its bounds), solve time; numbers = RMSE per output, solves by
   status, zero inputs, mean solve time (Path and Position RMSE went with the bicycle, #13).
3. **System generator** — the developer's pick for the next branch (2026-11-11). A plant
   declared as its ODE ẋ = f(x, u), its bounds and initial state, stepped by one integrator with
   the input held over T_s; today the coupled masses hand-code a forward-Euler step at T_s (the
   only plant since #13: LTI, inside DeePC's own model class — the generator is how a trusted
   nonlinear plant comes back). Moves numbers. Chosen: SciPy `solve_ivp` RK45 (Dormand–Prince),
   its tolerances dataset settings recorded in each run; two commits — the ODE declared with
   forward Euler kept (the safety net shows whether that alone moves a number), then the
   integrator switched and the net remade; tests against closed forms (the exact sampled
   coupled masses by matrix exponential).
4. Other platform gaps found on the way: the closed loop's numbers and `sim.metrics.
   compute_metrics` (what `tune.py` scores by) are two formulas for one set of metrics; no number
   card can plot against a dataset run's setting (e.g. an RMSE against T_s); `DeePCConfig`
   still holds bench settings; D6 and D8.
5. **Then the maths** — D4, D10, D12 (deferred by the developer on 2026-11-11: "a faulty
   pipeline, not platforming work"), D5, D7, D9, D11, D13–D17, all in `docs/project/failures.md`.

**Open for the developer** (asked, not answered):
- **The safety net is bit-identical on one computer only.** Made in the cloud container, it failed
  on the developer's computer (WSL) on `master` itself: the coupled masses' `sigma_norms` differed
  in the last bit (largest change 3e-30), the bicycle's loops by up to 6e-8 (2026-11-11,
  `git archive master` run there). PR #13 remade it on their computer, so the cloud may now fail
  it the other way. Which computer makes the net, or what it compares bit for bit, is theirs to
  decide (asked at the #13 hand-off, not answered).
- Seen in the Lab at the #13 hand-off, not explained: on the masses, sinusoidal, 3 s, defaults,
  e_x1 falls to about −1.5 m while F1 stays between −1.5 and +0.6 N, never at its 5 N bound; the
  same on `master`. For the maths phase.
- D7 (bicycle output bounds) and the `user_limit` count (measured on the bicycle) went with the
  bicycle; a `user_limit` solve on the coupled masses is unmeasured (one seen at λ_g = 20,
  sinusoidal, 1 s).
- Keep the SCS retry and the zero-input stand-in (PR #9 made both counted), or stop the run on a
  failed solve? (Moot if the rewrite's closed loop replaces it.)
- Found handing off `system-generator` (2026-11-11), not fixed: the Lab's address holds `shown`
  and `open` but not the piece, so a link always opens Closed loop; and switching the piece in
  the header logs a Plotly page error ("Cannot read properties of undefined (reading
  '_guiEditing')"), for Dataset as for System — the Lab code was not changed by that branch.
- The plant integrates by forward Euler at T_s itself, so a T_s sweep also changes the plant's
  discretisation (answered by item 3 if it lands).
