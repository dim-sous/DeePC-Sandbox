---
name: figure-conventions
description: Checklist for anything that draws or annotates a figure (a record's panel declaration, a Lab card, a still, the run.py report, a plant's plot hook). Figures describe; they never conclude.
when_to_use: Writing or editing plotting code (devkit/figure/, devkit/lab/, a piece's Record declaration, run.py, tune.py, a plant's plot_* method), annotating a chart, or choosing what a hand-off shows.
paths:
  - run.py
  - tune.py
  - tools/plants/**/*.py
  - devkit/figure/**/*.py
  - devkit/lab/**/*.py
  - devkit/harness/**/*.py
allowed-tools: Read Grep Glob Edit Write Bash
---

# Figures describe. They never conclude.

A claim drawn into a figure outlives the run that justified it and is read later as fact.
What a chart means belongs in the hand-off to the developer, where it can be argued with and
corrected. This is the figure face of "evidence over judgement" in `CLAUDE.md`.

## What may appear on a figure

- Axis labels, **with units**. One unit per axis.
- A legend naming **every** trace, when there is more than one thing to tell apart. One trace
  is named by its axis labels, so it gets no legend.
- Scaling that shows the data.
- A reference line or band **only when it is a quantity**: an input or rate bound, a
  reference trajectory, an expected value, a published number. It is named in the legend like
  any other trace. A bound names the quantity it bounds, with its unit (`[D7]`). The number is
  data; a judgement about it is not.

## What may not appear: check each one before you draw

- [ ] **No title.** Not even a neutral one. The panel's name names the figure.
- [ ] No verdict text anywhere: not in a label, a legend entry or a caption.
- [ ] No annotation telling the reader what to think, and no arrow pointing at the
      interesting part.
- [ ] No pass/fail wording, no "as expected", no "note that…".
- [ ] No colour or style chosen to say good or bad. Colours tell traces apart and nothing else.

If you catch yourself writing text that explains *why* the data looks the way it does, stop:
that sentence goes in the hand-off, not in the figure.

## Names

- A panel's name says what is drawn, never the activity or the module that made it. A name
  that claims more than the code delivers is a verdict, just like a title.
- Use the terms standard in the wider field (DeePC, Hankel, persistency of excitation, the
  literature's symbols: `Tini`, `N`, `λ_g`, `σ_y`), not only in this repo, and never give a
  word already in use here a second meaning.
- The words a Lab user reads (chart names, labels, legends, hovers, units) follow the
  developer's pattern (a capital first, no articles, acronyms and the standard mathematical
  terms), each listed in `docs/project/lab-words.md` and changed on their word.

## Every figure regenerates

One panel spec is drawn three ways (the Lab's card, a still, the numbers as CSV) from what a
record declares, as in BESS (`devkit/figure/`); `run.py`'s HTML report and `tune.py`'s are the
older figures, drawn from the run that wrote them. Either way: never edit an
image by hand, and never write a one-off script whose output cannot be made again.

## What the developer's gate needs

A branch that changes behaviour is handed over in the Lab (skill
`branch-handoff`, BESS `[B7]`). Choose the runs and charts so that together they:

1. reproduce a reference or known case,
2. show the effect or the trade the change introduces,
3. expose its limits.
