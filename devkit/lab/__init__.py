"""The Lab: one local page per piece, one screen (BESS #7, #10). The piece's runs on the left -- a
list to tick, filter and delete from, and the ticked runs compared column by column, numbers
first -- their cards on the right: a panel the records declare, or a number against the setting
that varies, each shaped by its runs control (:mod:`lab.shaping`). A draggable split between
the two, the form over the Config's groups in a drawer, and the terminal's own output in a
drawer along the bottom.

**It holds no logic of its own.** A UI that computes becomes a second implementation of the
thing it is supposed to show, and then the two disagree. So it does mechanical things only:

* **serves a page** on Panel (:mod:`lab.app`), on 127.0.0.1;
* **launches one command** as a subprocess -- ``deepc run``, the same line a person would type,
  printed as ``lab: <line>`` so it can be typed -- streaming and stopping the run;
* **reads** the registered Configs (their field declarations, through
  :func:`record.config.fields_of`), the record files on disk under the terminal's out-dir, and
  a shown record through its own class (:func:`lab.state.load`), which is reading too;
* **shows** what a card draws as the figure layer builds it (:mod:`figure.chart`, drawn by
  :func:`figure.twin.figure_of`): the same panel spec ``deepc draw`` draws its still from, so a
  chart on the page and the PNG are one drawing with no copy of its rules to drift;
* **says** what it did and what it read, in the terminal's words where they are the terminal's.

It never names a piece, never builds a Config, never runs one, never builds a trace of its own
(``tests/test_lab.py`` holds that by its import allowlist, by driving the page against a piece
that raises if it is run, and by checking every chart on the page is the figure layer's).
Every number on the page is the record's; every greyed control is its field's own gate or the
figure layer's refusal; every refusal is the words of whatever refused it.

Comparing runs is by name and by number: the shown runs share a card's chart -- on one pair of
axes, a row each, or pooled -- and each is a column in the table.
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv=None) -> int:
    from .cli import main
    return main(argv)
