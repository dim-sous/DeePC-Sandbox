"""The still's live twin: one :class:`figure.render.Panel` -> a plotly figure, as a plain dict.

It draws the panel spec the PNG renderer draws (lines, bands, references, dashes, colours, the
log flags, the x range, the labels), so a chart on the Lab's page and the still ``deepc draw``
writes are ONE drawing with no second set of rules to drift. The layer imports no plotly: a
figure is a dict, and the page's Plotly pane serialises it. Its layout -- the legend, the hover,
the heights, the rows labelled above -- is RCWS's ``render.interactive_figure``
(``feat/unequal-state-mixing``, 822b0cf), the developer's look.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np

from figure.render import BAND_ALPHA, SHORT, Panel, broken, has_legend
from figure.style import FONT, GRID, INK, INK_SECONDARY, LINE_WIDTH, MONO, REF_WIDTH, SURFACE

__all__ = ["figure_of", "ONE_PX", "MORE_ROW_PX"]

#: A one-row figure's height, and what each further row of a stacked figure adds (RCWS's
#: ``_ROW_PX``, ``_MORE_ROW_PX``).
ONE_PX, MORE_ROW_PX = 380, 150
#: The tallest the legend grows, as a share of the figure's height; past it, it scrolls. RCWS
#: ``render.LEGEND_HEIGHT``: unset, plotly let a horizontal legend take half the height -- on a
#: 380 px card over 8 runs the legend and its margin took the top 210 px.
LEGEND_HEIGHT = 0.3


def _rgba(hex_colour: str, alpha: float) -> str:
    n = int(hex_colour.lstrip("#"), 16)
    return f"rgba({n >> 16},{(n >> 8) & 255},{n & 255},{alpha})"


def _values(a: Any) -> list[Any]:
    out = np.asarray(a, dtype=float).tolist()
    return [None if v != v else v for v in out]           # NaN as a gap, which JSON can hold


def _axis(label: str) -> dict[str, Any]:
    # powers of ten as powers of ten, and only past 5 decades either way, the still's upper limit
    # (matplotlib's axes.formatter.limits, 6): plotly's SI prefixes wrote "100µ" on a
    # dimensionless axis in RCWS; its default of 3 decades wrote a 4 h run's seconds "0.5×10⁴";
    # no exponent at all wrote 2.6e-11 "2.0049999999999" (headless Chromium, 2026-10-27)
    return {"title": {"text": label, "font": {"size": 12, "color": INK}, "standoff": 8},
            "gridcolor": GRID, "zeroline": False, "linecolor": GRID, "tickfont": {"size": 11},
            "exponentformat": "power", "minexponent": 5}


def figure_of(panel: Panel, legend: bool = True) -> dict[str, Any]:
    """The figure for ``panel``: its rows on ONE bottom axis, so zooming one row zooms them all,
    each row's words written flat above it and the quantity once beside the rows; one legend
    entry per label, toggling that label in every row. ``legend=False`` leaves the legend out,
    for a view whose hover names every point."""
    n = len(panel.axes)
    stacked = n > 1
    shown = legend and has_legend(panel)
    # several points at one abscissa -- a sweep's runs at one setting -- hover the point under
    # the cursor: the box for one abscissa shows one point per curve
    crowded = any(np.unique(np.asarray(line.x, dtype=float)).size < len(line.x)
                  for ax in panel.axes for line in ax.lines)
    layout: dict[str, Any] = {
        "paper_bgcolor": SURFACE, "plot_bgcolor": SURFACE,
        "font": {"family": FONT, "size": 12, "color": INK_SECONDARY},
        "margin": {"l": 84 if stacked else 64, "r": 14, "t": 40 if shown else 16, "b": 44},
        # a path doubles back on its bottom axis, so one abscissa holds several of its points
        "hovermode": "closest" if crowded or panel.equal_axes else "x unified",
        # the hover writes every name whole: plotly's default cuts a name at 15 characters
        "hoverlabel": {"bgcolor": SURFACE, "bordercolor": GRID, "namelength": -1,
                       "font": {"family": MONO, "size": 11, "color": INK}},
        "showlegend": shown,
        "legend": {"orientation": "h", "x": 1, "y": 1.01, "xanchor": "right",
                   "yanchor": "bottom", "maxheight": LEGEND_HEIGHT, "bgcolor": "rgba(0,0,0,0)",
                   "font": {"size": 10, "color": INK_SECONDARY}},
        "annotations": [],
        "height": ONE_PX + MORE_ROW_PX * (n - 1),
        # what the panel is, in the record's own words, for whoever captions the figure
        "meta": {"panel": panel.name, "doc": panel.doc,
                 "rows": [ax.y_label for ax in panel.axes] if stacked else []},
    }
    data: list[dict[str, Any]] = []
    named: set[str] = set()

    def entry(label: str) -> bool:
        """A label is named in the legend where it first appears, in whichever row."""
        first = label not in named
        named.add(label)
        return first

    # every row has a gap above it, the top one included, so the first row's words sit under
    # the legend; the gap a share of the height that falls as rows are added (RCWS: a fixed
    # share left a row 26 px of a 2480 px card)
    gap = min(0.06, 0.12 / n) if stacked else 0.0
    h = (1.0 - n * gap) / n
    for i, ax in enumerate(panel.axes):
        axis_key, axis_ref = ("yaxis", "y") if i == 0 else (f"yaxis{i + 1}", f"y{i + 1}")
        top = 1.0 - gap - i * (h + gap)
        # rounded and clamped: plotly refuses a domain edge of -5e-17, which the last row's
        # arithmetic produces
        domain = [round(max(0.0, top - h), 6), round(min(1.0, top), 6)]
        layout[axis_key] = {**_axis("" if stacked else broken(ax.y_label, "<br>")),
                            "domain": domain, "anchor": "x",
                            "type": "log" if ax.log_y else "linear",
                            **({"scaleanchor": "x", "scaleratio": 1} if panel.equal_axes else {})}
        if stacked:
            layout["annotations"].append({
                "xref": "paper", "yref": "paper", "x": 0, "y": domain[1], "xanchor": "left",
                "yanchor": "bottom", "showarrow": False, "text": html.escape(ax.y_label),
                "font": {"size": 12, "color": INK}})
        for band in ax.bands:
            x = _values(band.x)
            lo = _values(np.asarray(band.lo, float) * np.ones(len(x)))
            hi = _values(np.asarray(band.hi, float) * np.ones(len(x)))
            data.append({"x": x, "y": lo, "yaxis": axis_ref, "mode": "lines",
                         "line": {"width": 0}, "hoverinfo": "skip", "showlegend": False,
                         "name": broken(band.label, "<br>"), "legendgroup": band.label})
            data.append({"x": x, "y": hi, "yaxis": axis_ref, "mode": "lines",
                         "line": {"width": 0}, "fill": "tonexty",
                         "fillcolor": _rgba(band.color, BAND_ALPHA), "hoverinfo": "skip",
                         "name": broken(band.label, "<br>"), "legendgroup": band.label,
                         "showlegend": entry(band.label)})
        xs = [v for line in ax.lines for v in line.drawn_x().tolist()]
        span = [min(xs), max(xs)] if xs else [0.0, 1.0]
        for ref in ax.refs:
            data.append({"x": span, "y": [ref.value, ref.value], "yaxis": axis_ref,
                         "mode": "lines", "line": {"width": REF_WIDTH, "color": ref.color,
                                                   "dash": ref.dash},
                         "name": broken(ref.label, "<br>"), "legendgroup": ref.label,
                         "showlegend": entry(ref.label), "hoverinfo": "skip"})
        for line in ax.lines:
            marked = bool(line.texts) or len(line.x) < SHORT
            trace = {"x": _values(line.drawn_x()), "y": _values(line.y), "yaxis": axis_ref,
                     "mode": "markers" if line.points else ("lines+markers" if marked else "lines"),
                     "line": {"width": LINE_WIDTH, "color": line.color, "dash": line.dash},
                     "marker": {"size": 7 if line.texts else 5, "color": line.color},
                     "name": broken(line.label, "<br>"), "legendgroup": line.label,
                     "showlegend": entry(line.label),
                     # a path's point is both its coordinates: no unified header gives the x
                     "hovertemplate": "%{x:.6g}, %{y:.6g}" if panel.equal_axes else "%{y:.6g}"}
            if line.texts:
                # the point's run, its value, and beside them the curve's name -- what its colour
                # stands for, read there on a card that draws no legend (RCWS)
                trace["text"] = list(line.texts)
                trace["hovertemplate"] = "%{text}: %{y:.6g}"
            data.append(trace)
    if stacked:
        layout["annotations"].append({
            "xref": "paper", "yref": "paper", "x": 0, "y": 0.5, "xanchor": "right",
            "yanchor": "middle", "xshift": -60, "textangle": -90, "showarrow": False,
            "text": html.escape(panel.quantity), "font": {"size": 12, "color": INK}})
    xaxis: dict[str, Any] = {**_axis(panel.x_label), "anchor": f"y{n}" if stacked else "y",
                             "domain": [0, 1], "type": "log" if panel.log_x else "linear"}
    if panel.x_ticks:
        xaxis.update({"tickvals": [p for p, _ in panel.x_ticks],
                      "ticktext": [w for _, w in panel.x_ticks]})
    if panel.x_range is not None:
        # plotly holds a log axis's range as powers of ten
        rng = np.log10(panel.x_range).tolist() if panel.log_x else list(panel.x_range)
        xaxis.update({"range": rng, "autorange": False})
    layout["xaxis"] = xaxis
    return {"data": data, "layout": layout}
