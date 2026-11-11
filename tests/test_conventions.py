"""Conventions the repository holds about its own documents.

Advisory rules live in CLAUDE.md; a rule that can be checked lives in a test, and the test's
docstring carries its incident tag from ``docs/project/failures.md``.
"""

import re
from pathlib import Path

import pytest

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


METHODS_MD = ROOT / "METHODS.md"
README_MD = ROOT / "README.md"
FAILURES_MD = ROOT / "docs" / "project" / "failures.md"
#: Folders a document may name that a fresh clone does not hold: made by use, or git-ignored.
NOT_IN_GIT = ("references/", "results/", ".venv/")
#: A backticked repository path: one with a folder (``devkit/sim/simulation.py``, ``tests/reference/``),
#: or a file with a ``:line`` pointer (``tune.py:40``), the pointer optional on the first and a
#: range allowed. A bare file name with no folder and no pointer is not checked: in a table row
#: it is named relative to the folder the row is about.
_PATH = re.compile(r"`(?P<path>[\w.][\w./-]*/[\w./-]*|[\w.-]+\.\w+(?=:))"
                   r"(?::(?P<first>\d+)(?:-(?P<last>\d+))?)?`")


def _prose(text: str) -> str:
    """``text`` without its code blocks, inline code and maths, where brackets are not citations."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    return re.sub(r"\$[^$\n]*\$", "", text)


def _cited(prose: str) -> set[int]:
    """The reference numbers ``prose`` cites: ``[1]``, ``[1, 4]``, ``[2–3]`` (a range)."""
    out: set[int] = set()
    for group in re.findall(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", prose):
        for part in re.split(r"\s*,\s*", group):
            ends = [int(n) for n in re.split(r"\s*[–-]\s*", part)]
            out.update(range(ends[0], ends[-1] + 1))
    return out


def test_every_citation_in_methods_md_is_a_reference_and_every_reference_is_cited():
    """A numbered citation with no entry, or an entry nothing cites, is a claim about the
    literature nobody can follow. The rule: literature-grounded, a source cited where it is used
    (CLAUDE.md); the practice is Furuta-Pendulum's METHODS.md."""
    body, _, listed = METHODS_MD.read_text().partition("\n## References\n")
    cited = _cited(_prose(body))
    entries = {int(n) for n in re.findall(r"^(\d+)\. ", listed, flags=re.M)}
    assert cited == entries, (f"cited but not listed: {sorted(cited - entries)}; "
                              f"listed but not cited: {sorted(entries - cited)}")


@pytest.mark.parametrize("document", [README_MD, METHODS_MD, FAILURES_MD], ids=lambda p: p.name)
def test_every_path_a_document_names_exists(document: Path):
    """Every repository path a document names in backticks exists (see ``_PATH`` for which), and
    a ``path:line`` pointer falls inside its file: after a move the documents are searched for
    the old names (RCWS #159), and this is the half of that a test can hold. Whether a pointer
    still points at the code it describes is not checked: that needs a reader, and the documents
    name the commit their pointers were read at."""
    missing = []
    for match in _PATH.finditer(document.read_text()):
        path, first, last = match["path"], match["first"], match["last"]
        if path.startswith(NOT_IN_GIT):
            continue
        target = ROOT / path
        if not target.exists():
            missing.append(path)
        elif first and len(target.read_text().splitlines()) < int(last or first):
            missing.append(f"{path}:{first}{'-' + last if last else ''} (past the end of the file)")
    assert not missing, f"{document.name} names what the repository does not hold: {missing}"
