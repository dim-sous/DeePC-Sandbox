"""A draggable split between two panes -- the one piece of layout Panel does not ship."""

from __future__ import annotations

import param
from panel.custom import Child, JSComponent

from .skin import SPLIT

DEFAULT_PX = 400


class Split(JSComponent):
    """``left`` | handle | ``right``. Dragging the handle sets ``left_px`` (the left pane's
    width; not ``width``, which is the component's own layout width), double-clicking it resets
    it; the page reads and sets the same parameter. A drag ends with a window ``resize`` event,
    so charts that fit their container refit."""

    left = Child()
    right = Child()
    left_px = param.Integer(default=DEFAULT_PX, bounds=(260, None),
                            doc="the left pane's width in px")

    _stylesheets = [SPLIT]
    _esm = """
    export function render({ model }) {
      const box = document.createElement('div'); box.className = 'split';
      const left = model.get_child('left'), right = model.get_child('right');
      const handle = document.createElement('div'); handle.className = 'handle';
      handle.title = 'drag to resize · double-click to reset';
      const apply = () => { box.style.gridTemplateColumns = model.left_px + 'px 6px minmax(0,1fr)'; };
      let drag = false;
      handle.addEventListener('pointerdown', e => {
        drag = true; handle.classList.add('dragging'); handle.setPointerCapture(e.pointerId);
        e.preventDefault(); });
      handle.addEventListener('pointermove', e => {
        if (!drag) return;
        const rect = box.getBoundingClientRect();
        // The low bound LAST: in a container under ~371px, seven tenths of it is under the
        // 260 the parameter admits, and clamping high-then-low there sets a width it rejects.
        const w = Math.round(Math.max(Math.min(e.clientX - rect.left, rect.width * 0.7), 260));
        model.left_px = w; apply(); });
      const done = () => { if (!drag) return; drag = false; handle.classList.remove('dragging');
        window.dispatchEvent(new Event('resize')); };
      handle.addEventListener('pointerup', done);
      handle.addEventListener('pointercancel', done);
      handle.addEventListener('dblclick', () => { model.left_px = %d; apply();
        window.dispatchEvent(new Event('resize')); });
      model.on('left_px', apply);
      apply();
      box.append(left, handle, right);
      return box;
    }
    """ % DEFAULT_PX
