"""The figure layer: stills over one or more Records, and each panel's live twin for the Lab's
page, drawn from ONE panel spec (``render.Panel``) that :mod:`figure.chart` builds from what the
records declare -- a ``Series`` with its runs overlaid, split or pooled, or a summary number
against a setting. It never names a piece: a record that starts carrying a new panel gains a
figure with no edit here. Figures DESCRIBE: axis labels with units (one unit per axis), a legend
only when there is more than one label, no title. Every save goes through ``style.save_figure``.
"""

import matplotlib

matplotlib.use("Agg")  # headless; must precede any pyplot import

from figure.draw import draw, draw_one, draw_sweep  # noqa: E402

__all__ = ["draw", "draw_one", "draw_sweep"]
