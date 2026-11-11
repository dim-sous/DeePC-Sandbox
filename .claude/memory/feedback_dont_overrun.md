---
name: dont-overrun
description: The developer lost patience with long waits (2026-11-11); run the affected tests while fixing, the whole suite once per branch
metadata:
  type: feedback
  modified: 2026-11-11
---

2026-11-11, during the smaller-Lab branch: "this takes way too long, why? it seems weird". On
that branch I had run the whole suite (about 2.5 minutes each) four times, plus partial runs.

**How to apply:**
- While fixing, run only the tests the change touches; run the whole suite once, at the end of
  the branch (it is the gate) — again only if code changed after it.
- Measure before explaining a slowness: `uv run pytest --durations=0`, grouped by file.
- Say where things stand between long steps; never several silent minutes.

BESS's note of the same kind (`feedback_dont_overrun.md` there) came from "wtf is taking so long
again????" on 2026-10-06.
