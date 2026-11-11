"""A system typed as rows, x' = f(x, u, θ), y = h(x, θ), with its θ values (``system.parse``),
and the one integrator (``system.integrate.hold``). The initial state, the input bounds and the
sampling time are the open-loop simulation's, not the system's (docs/project/design.md)."""

from __future__ import annotations

from system.integrate import ATOL, RTOL, hold
from system.parse import Parsed, parse

__all__ = ["Parsed", "parse", "hold", "RTOL", "ATOL"]
