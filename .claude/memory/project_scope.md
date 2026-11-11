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

Earlier choices, still standing: the BESS workflow; SI units for the coupled masses (assumed —
the code states none); Lab words follow their pattern rather than being asked one by one; the
papers stay out of git (`references/`, the repository is public); MIT licence.
