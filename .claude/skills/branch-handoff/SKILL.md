---
name: branch-handoff
description: Take a finished branch to the developer's gate and stop there — the whole suite run locally, the code review run, the runs made, then the hand-off in the Lab (the address with the runs and charts open, what to click, what each chart shows), or plainly "nothing to drive" when the branch has nothing to judge. Nothing is committed, pushed or merged until the developer says so.
when_to_use: A branch's code and tests are finished; before any git write; when asked to "wrap up", "hand off", "finish the branch", or "is this ready to merge".
argument-hint: "[branch topic]"
allowed-tools: Read Grep Glob Bash
---

# The developer's gate

The developer judges maths and controller performance, never code (BESS `[B6]`). This skill
takes a branch to the edge of that gate and stops there.

**No commit, no push, no PR, no merge** until the developer says so in the current turn.

## 1. Code and tests are finished

Every part of the scope, not the easy parts. If something is blocked, finish the rest and say
plainly what was left out and why.

## 2. Run the suite

```
uv run pytest
```

Run the whole suite, locally: it is the PR gate. Report the real result and its wall time; if
a test fails, say so and show its output. A branch meant to move no number keeps the safety
net (`tests/test_reference.py`) green; a branch meant to move numbers remakes it
(`uv run python -m tests.reference.make`) and says which cases moved and why.

## 3. Run the code review yourself

`/code-review` over the branch, then fix what it finds, or say why a finding stays. The PR
says the review was run, how many findings it made, and how many were fixed (BESS `[B6]`).

## 4. Make the runs the developer will look at

`uv run deepc run <piece> [--set k=v] [--sweep k=a,b] --name <name>`, which writes
`results/<piece>/<name>.json` (`devkit/README.md`), or the same from the Lab's form. Choose the
runs so that together they reproduce a reference or known case, show what the branch changes,
and expose its limits. A refactor proves itself by NOT
changing the numbers (BESS `[B4]`).

## 5. Name the conditions every number came from

For each number you hand over, give the conditions in full: the plant, the scenario, the seed,
the duration, and every setting off its default (`[D1]`). A default config is a posture like
any other, not a neutral one. One posture at one seed is that posture at that seed, not a
capability.

## 6. Check the Lab shows it

Open the address you will hand over in headless Chromium and check that every chart named in
it draws and that no error shows on the page; `tests/test_lab_browser.py` shows how to serve the
page and drive it. If the developer already has a Lab running, the new runs appear in it after
Refresh.

## 7. Pause and hand off, in the Lab

Give the developer, in plain words (skill `plain`):

- **What changed**, and which gap it closes.
- **Where to look in the Lab:** the address with the runs and charts open
  (`http://127.0.0.1:8765/?shown=a,b&open=chart-1,chart-2`), and what to click. On their
  computer: `uv run lab` (`--no-open` prints the address instead of opening a browser); from a
  cloud session, the page served there and its port reported to the app.
- **What each chart shows, described and not concluded**: what is on each axis, what each
  trace is, and what was run to make it. Never hand over an image file to open (BESS `[B7]`).
- If the branch has nothing the developer can judge, say so plainly (BESS `[B6]`).
- **The suite result** and **the code review's result**, stated plainly.
- **Limits:** what this does not cover, what is still open, what you are unsure of, and
  anything you assumed rather than checked.

Then stop and wait.

## 8. After the developer says so, and only then

1. Write down in `docs/project/failures.md` any incident that produced a rule.
2. Commit, push, open the PR: its body says what was handed over and how.
3. Merge with a merge commit, never squash or rebase, and only after the suite is green
   locally. Delete the branch.

Each git write needs the developer's word in the turn it happens. An earlier approval does
not carry over.
