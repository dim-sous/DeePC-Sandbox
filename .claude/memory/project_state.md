---
name: project-state
description: Where the DeePC-Sandbox rebuild stands — what is merged, what comes next, what is still open for the developer
metadata:
  type: project
  modified: 2026-11-11
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

**Next, in the order agreed** (the developer picks; propose, then plan mode, then build):
1. **Test speed** — the suite took 148 s on 2026-11-11, measured with `--durations`: about 70 %
   simulating closed loops (the same loops rebuilt in several test files), 22 % the browser
   tests. Proposed: pytest-xdist `-n auto` as RCWS's `pyproject.toml` does (4 cores in the cloud
   container), the tests sharing a port or the Lab's one-command lock kept on one worker; and
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
  failed solve?
- The plant integrates by forward Euler at T_s itself, so a T_s sweep also changes the plant's
  discretisation (answered by item 3 if it lands).
