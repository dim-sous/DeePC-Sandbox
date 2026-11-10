"""Conventions the repository holds about its own documents.

Advisory rules live in CLAUDE.md; a rule that can be checked lives in a test, and the test's
docstring carries its incident tag from ``docs/project/failures.md``.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = ROOT / "CLAUDE.md"

#: CLAUDE.md's heft budget, in lines: BESS's. The file landed at 126 when rewritten to advisory
#: rules + pointers on 2026-11-10. Over budget is not a licence to raise the number.
CLAUDE_MD_BUDGET = 140


def test_claude_md_stays_within_its_heft_budget():
    """BESS [B5]: its old CLAUDE.md carried a ten-metric list no code produced, and nothing
    pushed back on prose accreting in the one file every session loads. This repository's old
    CLAUDE.md carried five "lessons learned" no record backs, one of them measured through
    [D1]. The rule: the RULE and its pointer land in CLAUDE.md, the DESCRIPTION in the README,
    the NUMBER in a record — so the fix for being over budget is almost always a description or
    a list moved out, never a bigger budget."""
    lines = len(CLAUDE_MD.read_text().splitlines())
    assert lines <= CLAUDE_MD_BUDGET, (
        f"CLAUDE.md is {lines} lines, over its {CLAUDE_MD_BUDGET}-line budget by "
        f"{lines - CLAUDE_MD_BUDGET}. Move a description to the README or a number to a record; "
        f"do not raise the budget."
    )
