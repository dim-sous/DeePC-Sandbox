# DeePC Sandbox — data-enabled predictive control on simulated plants

A DeePC controller (Coulson, Lygeros & Dörfler 2019) run on two simulated plants, a kinematic
bicycle and a coupled two-mass spring-damper. It is being rebuilt as BESS and RCWS were: first
the scaffolding (rules, safety net, records, a Lab) to expose the gaps and bugs in today's code,
then the fixes. What it is and how it runs: `README.md`. This file holds the working rules.

**Only advisory rules live here** — the ones no test can catch. A checked rule lives in the
test that checks it, and that test's docstring carries its incident tag. Each rule points at
what holds it: an incident in `docs/project/failures.md` (`[D1]`–`[D17]`; a rule without its
incident degrades into a slogan), a sibling repository's incident (BESS `[B…]`, RCWS `[A…]`),
or a test. When in doubt, the rules win.

## Commands

- Python through uv only, 3.13 as pinned: `uv run pytest …`, `uv run python …` — never a venv
  python. For parallel jobs `uv sync` once, then `uv run --no-sync`.
- Entry points, until the terminal and the Lab land: `uv run python run.py [--plant …]
  [--scenario …] [flags]` (one HTML report and one metrics file under `results/<tag>/`;
  `--help` lists the flags). `tune.py` is set aside: its `Ts` search runs through `[D1]`.
- The suite: `uv run pytest`, whole — it runs LOCALLY and IS the PR gate. Its safety net,
  `tests/reference/`, is remade only by `uv run python -m tests.reference.make`, on a branch
  that means to move the numbers, and that branch's PR says so.

## Workflow

- One branch per gap (group similar ones), landed through a GitHub PR — merge commit, never
  squash/rebase. Merge only after the suite is green LOCALLY. Branch deleted on merge; the
  clean read of `master` is `git log --oneline --first-parent`.
- **Pause before git on something the developer can JUDGE:** run the code review yourself
  (`/code-review`; say so in the PR), then hand off IN THE LAB — the address with its runs and
  charts open, what to click, what each chart shows; never an image file — or say plainly there
  is nothing to drive (true until the Lab lands). No git write until they say so, each time; an
  earlier approval does not carry over. Skill: `branch-handoff`. (BESS `[B6, B7]`)
- Propose, then plan mode, then build: a short what/why/risks/dependencies note before any
  code, plan mode for anything non-trivial. One gap per branch; the smallest change that
  closes it, its cost in lines and compute time named; no abstraction for a future no branch
  needs yet.
- Short supervised stretches; ask before ANY multi-agent fan-out, every time (RCWS `[A12]`).
  "Investigate X" means diagnose-and-report. Asked to finish or resume something, run exactly
  the missing pieces — never a widened scope.
- A refactor proves itself by NOT changing the numbers: the safety net, bit for bit; a test
  that checks a key exists proves nothing. (BESS `[B4]`)
- Name the conditions every number came from — the plant, the scenario, the seed, the
  duration, and every setting off its default, stated in full; a default config is a posture,
  not a neutral one; a run at one condition is that condition, not a capability. `[D1]`
- Where a change is written down: the RULE here (or in its test), the DESCRIPTION in the
  README, the METHOD in `METHODS.md` (changed when a method changes, every citation marked
  until checked against its source), the INCIDENT in `failures.md`. A measured number belongs
  at its point of use with what was observed, or in the record that regenerates it — never
  transcribed into prose, where it goes stale unread. (BESS `[B3, B5]`)
- Heft is enforced: this file by `test_claude_md_stays_within_its_heft_budget`.
- Explanations start from what the developer can see; the maths keeps its terms. Skill `plain`.
- Git is the only sync between computers; ignore only regenerable or machine-specific files,
  and `references/` (the papers): the repository is public.

## Architecture

Layout is where files live; architecture is what depends on what — keep the two words apart.
Today: `control/` is the DeePC controller (Hankel matrices, the sparse QP, the
persistent-excitation check, the noise-adaptive and online-Hankel variants); `plants/` the
simulated plants, each with its data-collection recipe; `sim/` data collection, scenarios and
the closed loop; `run.py` the command, the metrics and the report. The target, BESS's: three
roots, one direction — `src/` the controller, `tools/` the plants, `devkit/` the bench
(records, figures, the terminal, the Lab) — imports running `src ← tools ← devkit`, held by a
test once it lands.

The judgment calls no test can make yet:

- A change in A must not require a change in B — when a change forces an edit somewhere it has
  no business reaching, the coupling is the defect, not the edit. `[D5]`
- The controller names no plant: it receives values (data, outputs, constraints), never a plant
  class. Today `control/controller.py` imports `plants.base.Constraints`; the move removes it.
- One block, one package; one file per variant, and the block's registry is the only place
  that knows which exist. Code nothing calls goes. `[D8]`
- A setting the run did not use is refused or recorded as unused, never accepted. `[D1, D11]`
- One quantity, one formula: the output the training data holds and the one the loop gives the
  controller come from one function, or the divergence is declared where it lives. `[D4, D10, D12]`
- A number is stated once, where it is used, with its source; everything else reads it.
- Any value a block computes is surfaced with one line — kept by the record, plotted by the
  Lab, with no other edit (RCWS `dd18796` took ten files).
- After any move, search code, docs and settings for the old names (RCWS #159).
- The words a Lab user reads — labels, hovers, chart names, legends — are the developer's,
  asked instance by instance, never decided (RCWS #96).

## Conventions & invariants

Break these and the results are worthless: evidence over judgement · no cheating · independent
truth. Break these and the work needs redoing: realism kept · literature-grounded · compute
cost named. Re-read this section before writing a COMMENT or README line that says what the
code does, and before adding any ERROR HANDLING, fallback or guard.

**Evidence over judgement — you are a programming assistant, NOT a scientist.** Never assert a
physical, mathematical or engineering explanation as established. The permitted moves:
reproduce the cited reference and locate the difference; quote what a named source says; or
say "I don't know — here is the experiment", then run it. One rule, four faces:

- In prose — README, PR bodies, commit messages: a claim about the code is checked against the
  source the day it is written. The README's "claims from before the rebuild" stay unverified
  until a record re-measures them. `[D1]`
- In code — a comment states what the code DOES, checked against the lines below it;
  identifiers name the condition, never its cause. `[D6]`
- In figures — they DESCRIBE, never conclude (skill `figure-conventions`): axis labels with
  units (one unit per axis), a legend naming every trace when more than one, nothing else, no
  title. A bound names the quantity it bounds, with its unit. `[D7]`
- In error handling — fail LOUD: no swallowed exceptions, no stand-in answer for one the code
  failed to compute. A solver status is counted by its own name in the record, and a fallback
  is counted where it happens. `[D2, D3]`

The comment rule — four kinds, nothing else: (1) function/parameter/return doc with units;
(2) a citation (source + page or equation); (3) a `# MISMATCH` flag where the code knowingly
departs from the plant or the cited method, saying what it does instead; (4) a decision with
its observed consequence. State what was observed, never what it means.

- No cheating — name it *data leakage*: the plant's state generates its outputs, the scores and
  the plots; it reaches the controller only as the output the plant reports. `[D4]`
- Realism kept: the plant is never tuned to make the controller look right; its step, noise
  and limits are the plant's, set apart from the controller's settings. `[D1]`
- Independent truth (the *inverse crime*): a controller scored on the plant, seed and noise its
  data came from is scored on its own data source. Say so in every such number; prefer
  conditions the data never saw.
- Compute cost is part of the result: solve time and its spread are summary numbers in every
  record, not a footnote.
- Literature-grounded: when the thing exists in the literature (regularised DeePC, online
  Hankel updates, persistency of excitation), find the established method BEFORE designing;
  cite it where it is used; a departure from it is declared. `[D9, D13–D17]`
- Every guard makes a failure less legible: a new clip, fallback, threshold or relaxation must
  earn its place, states its source, and defaults to reporting, not acting. `[D3, D5]`
