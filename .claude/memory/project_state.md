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
the fixes. All work so far ran in one cloud session; this note is its hand-over.

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

**Next, in the order agreed** (the developer picks; propose, then plan mode, then build):
1. **Test speed** — the suite took 148 s on 2026-11-11, measured with `--durations`: about 70 %
   simulating closed loops (the same loops rebuilt in several test files), 22 % the browser
   tests. Proposed: pytest-xdist `-n auto` as RCWS's `pyproject.toml` does (4 cores in the cloud
   container), the tests sharing a port or the Lab's one-command lock kept on one worker; and
   the shared runs built once per suite. No number moves.
2. **Fewer charts and numbers per record** — proposed, not yet confirmed in detail: charts =
   each output, each input (with its bounds), Path, solve time; numbers = RMSE per output,
   Position RMSE, solves by status, zero inputs, mean solve time.
3. **System generator** — a plant declared as its ODE ẋ = f(x, u), y = h(x), its bounds and
   initial state, stepped by one integrator with the input held over T_s; today each plant
   hand-codes a forward-Euler step at T_s. Moves numbers.
4. Other platform gaps found on the way: the closed loop's numbers and `sim.metrics.
   compute_metrics` (what `tune.py` scores by) are two formulas for one set of metrics; no number
   card can plot against a dataset run's setting (e.g. Position RMSE against T_s); `DeePCConfig`
   still holds bench settings; D6 and D8.
5. **Then the maths** — D4, D10, D12 (deferred by the developer on 2026-11-11: "a faulty
   pipeline, not platforming work"), D5, D7, D9, D11, D13–D17, all in `docs/project/failures.md`.

**Open for the developer** (asked, not answered):
- D7: were output bounds meant on the bicycle's speed, not on its speed error?
- A `user_limit` solve (OSQP at its iteration cap) is applied as an answer and not flagged
  (bicycle, λ_y = 10¹², 10 s: 93 of 100 solves) — count it as no answer?
- Keep the SCS retry and the zero-input stand-in (PR #9 made both counted), or stop the run on a
  failed solve?
- Both plants integrate by forward Euler at T_s itself, so a T_s sweep also changes the plant's
  discretisation (answered by item 3 if it lands).
