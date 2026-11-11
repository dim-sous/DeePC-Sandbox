"""The compare table: the shown runs column by column (:func:`lab.view.compare_rows`), and what
a row of it does -- a number opens the card that draws it against the setting that varies (RCWS
``lab/compare.py``)."""

from __future__ import annotations

from typing import Any, Callable, Optional

import param
from panel.custom import JSComponent

from .skin import TABLE


class CompareTable(JSComponent):
    """``heads`` (the shown runs, in the order they were ticked), ``corner`` (the words over the
    terms) and ``rows`` (``kind``, ``key``, ``term``, ``hover``, ``differs``, ``open`` and the
    ``values`` per run, written out) in. A click on a ``number`` row reaches :attr:`on_number`
    with its key as a MESSAGE, so the same row clicked twice reaches it twice. The table is as
    wide as its pane, whatever it holds (RCWS's): the words take :data:`skin.TERMS_SHARE` per
    cent of it, the runs share the rest equally, a long name or value wraps inside its own
    column, and when the runs' columns would fall under :data:`skin.RUN_MIN_PX` it scrolls
    sideways with the words held in place."""

    heads = param.List(default=[], item_type=str)
    corner = param.String(default="")
    rows = param.List(default=[], item_type=dict)
    empty = param.String(default="", doc="what the table says when no run is shown")

    on_number: Optional[Callable[[str], None]] = None

    _stylesheets = [TABLE]
    _esm = """
    export function render({ model }) {
      const root = document.createElement('div');
      const el = (tag, cls, text) => { const e = document.createElement(tag);
        if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
      const draw = () => {
        root.replaceChildren();
        if (!model.heads.length) { root.append(el('div', 'empty', model.empty)); return; }
        const box = el('div', 'box'), table = el('table', 'cmp'), cols = el('colgroup', ''),
              head = el('tr', '');
        table.style.setProperty('--runs', model.heads.length);
        cols.append(el('col', 'terms'));
        for (const _ of model.heads) cols.append(el('col', ''));
        head.append(el('th', '', model.corner));
        for (const name of model.heads) head.append(el('th', '', name));
        const thead = el('thead', ''); thead.append(head);
        const body = el('tbody', '');
        for (const r of model.rows) {
          if (r.kind === 'head') {
            const tr = el('tr', 'head'), td = el('td', '', r.term);
            td.colSpan = model.heads.length + 1; tr.append(td); body.append(tr); continue;
          }
          const click = r.kind === 'number';
          const tr = el('tr', [r.differs ? 'differs' : '', click ? 'click' : '',
                               r.open ? 'open' : ''].join(' ').trim());
          const term = el('td', '', r.term);
          term.title = (r.hover || '') + (click ? ' · Click to plot it against a setting' : '');
          if (click) term.addEventListener('click', () => model.send_msg({ number: r.key }));
          tr.append(term);
          for (const v of r.values) tr.append(el('td', '', v));
          body.append(tr);
        }
        table.append(cols, thead, body); box.append(table); root.append(box);
      };
      for (const p of ['heads', 'corner', 'rows', 'empty']) model.on(p, draw);
      draw();
      return root;
    }
    """

    def _handle_msg(self, msg: Any) -> None:
        if isinstance(msg, dict) and isinstance(msg.get("number"), str) \
                and self.on_number is not None:
            self.on_number(msg["number"])
