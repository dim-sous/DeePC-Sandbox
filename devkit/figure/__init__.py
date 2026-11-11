"""The figure layer: stills over one or more Records, drawn from ONE panel spec
(``render.Panel``) that :mod:`figure.chart` builds from what the records declare -- a ``Series``,
or a summary number against a setting. It never names a piece: a record that starts declaring a
new panel gains a figure with no edit here. Figures DESCRIBE: axis labels with units (one unit
per axis), a legend only when there is more than one label, no title. Every save goes through
``style.save_figure``. BESS's ``devkit/figure``, trimmed.
"""

import matplotlib

matplotlib.use("Agg")  # headless; must precede any pyplot import

from figure.draw import draw, draw_sweep  # noqa: E402

__all__ = ["draw", "draw_sweep"]
