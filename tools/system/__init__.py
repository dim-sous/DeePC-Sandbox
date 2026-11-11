"""A system typed as rows, x' = f(x, u, d, θ), y = h(x, θ), with its θ values (``system.parse``),
the one integrator (``system.integrate.hold``) and the plant's sampling step with its actuator
clip (``system.step.step``). The initial state, the input bounds and the sampling time are the
open-loop simulation's settings, not the system's (docs/project/design.md)."""

from __future__ import annotations

from system.integrate import ATOL, RTOL, hold
from system.parse import Parsed, parse
from system.step import step

__all__ = ["Parsed", "parse", "hold", "step", "RTOL", "ATOL"]
