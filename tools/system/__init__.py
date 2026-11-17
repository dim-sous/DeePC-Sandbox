"""A system typed as rows, x' = f(x, u, d, θ), y = h(x, θ), with its θ values (``system.parse``),
the one integrator (``system.integrate.hold``), the plant's sample -- its output measured
(``system.step.measure``), then its step with the actuator clip (``system.step.step``) -- and a
signal typed as rows in t, such as a reference r(t) (``system.parse.parse_signal``). The initial
state, the input bounds and the sampling time are the open-loop simulation's settings, not the
system's (docs/project/design.md)."""

from __future__ import annotations

from system.integrate import ATOL, RTOL, hold
from system.linearize import linearize
from system.parse import Parsed, parse, parse_signal
from system.step import measure, step

__all__ = ["Parsed", "parse", "parse_signal", "hold", "step", "linearize", "measure", "RTOL", "ATOL"]
