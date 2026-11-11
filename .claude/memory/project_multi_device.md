---
name: multi-device
description: The developer works from a cloud session and from VS Code on their computer; git is the only sync, and these notes carry the context between sessions
metadata:
  type: project
  modified: 2026-11-11
---

From 2026-11-10 to 11 all the work ran in one Claude Code cloud session (its branch was named
`ccr-a2b958a4-genwi4` and restarted from `master` after every merge); the developer followed it
from a phone and from a computer that did not hold the repository. On 2026-11-11 they moved to VS
Code on their own computer, noting that "the entire context of the refactoring is localized
here, there is nothing in the repo to guide you in a different session" — so this folder.

**A new session starts here**: CLAUDE.md points to `.claude/memory/MEMORY.md`. Keep these notes
current at the end of every branch and commit them with it. As BESS does, a computer may also
point Claude Code's own notes here: `.claude/settings.local.json` (git-ignored) with
`{"autoMemoryDirectory": "<this repository>/.claude/memory"}`.

**On their computer after a pull**: `uv sync` when packages moved; `uv run playwright install
chromium` once; `uv run deepc run dataset`, `uv run deepc run closed_loop --set sim_duration=10`,
`uv run lab`. Records saved before a branch that changed a record's fields show as stale in the
Lab: re-run them. A computer that held the code before the layout move (PR #4) may keep git-ignored
`control/`, `plants/`, `sim/` folders at the root holding only `__pycache__`: they shadow the real
packages (`import plants` finds an empty namespace) and every test fails to import. On the
developer's computer they were deleted, on their word, 2026-11-11.
