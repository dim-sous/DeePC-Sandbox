---
name: project-scope
description: The developer's scope for DeePC-Sandbox (2026-11-11) — small, three parts, a small Lab, platform before maths
metadata:
  type: project
  modified: 2026-11-11
---

The developer, 2026-11-11: "Contrary to the BESS and RCWS project, this project scope is much
smaller and simpler: closed loop simulation using deepc under different configurations. The only
extra thing would be a data generator and maybe a plant/system generator. Define with the later
some ODEs (or smth like that) and simulate it with the former. Then the main component runs
deepc. In that context, I would expect this project to be much smaller that the other two and the
same holds for the lab as well (less numbers, less charts, less buttons and less options in
general)."

So the shape is three parts: a **system generator** (a plant from its ODEs), a **data generator**
(the `dataset` piece), and **DeePC in closed loop** (the `closed_loop` piece). Measured the same
day: the bench (`devkit/`) was 3.8 times the code it studies (`src/` + `tools/`), BESS's 2.4.

**The Lab cut** (their choices, AskUserQuestion, merged as PR #11): kept Split and Mean/RMS on a
chart card and the run-list housekeeping (filter, delete, divider, folded groups, Compare's
None/Differing/All); removed the axis controls (log, dots/lines, X choice, colour by), Save, the
saved stills and the Table view. A chart card's one control is Runs; a number card has X and Runs.

**Platform first, maths after** (2026-11-11, on D4/D10/D12): "these sound more like a faulty
pipeline, not platforming work. The math, the pipeline, reporting, charts, numbers, tools etc
will be reexamined once the lab is finished". Fixes to what the controller or the plants compute
wait until the platform branches are done ([[project-state]] lists them).

**Three directions, 2026-11-11 evening** (after PR #13, starting the system generator; "right
after that"):
1. "I dont know why i decided to track the error instead of the state but we are reverting to
   the state" — the controller tracks the state, not errors against a zero reference (D15).
2. "Since we are generating systems through the lab lets keep things as generic as possible":
   (a) **no units anywhere**; (b) **inputs (manipulated variables) are u's, outputs
   (measurements) are y's, true states are x's** — "the names everywhere (axis labels, chart
   buttons, EVERYWHERE)". This replaces the plant's own channel names (F1, e_x1, …) and the SI
   units assumed for the masses; it overrides CLAUDE.md's figure rule "axis labels with units"
   and its test, which a branch must change on their word.
3. Systems are generated **through the Lab** (their words; what that means is asked).

Earlier choices, still standing: the BESS workflow; Lab words otherwise follow their pattern
rather than being asked one by one; the papers stay out of git (`references/`, the repository
is public); MIT licence. (SI units for the masses: superseded by 2a.)
