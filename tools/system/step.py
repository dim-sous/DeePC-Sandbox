"""One sampling step of the plant: the input clipped to the actuator bounds and, with the
disturbance, held over T_s (zero-order hold), and the state integrated (``integrate.hold``). The
clip is the plant's own limit; which channels it changed is returned, so the caller counts it
(CLAUDE.md: a guard defaults to reporting)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from system.integrate import ATOL, RTOL, hold

__all__ = ["step"]


def step(f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], x: np.ndarray,
         u: np.ndarray, d: np.ndarray, Ts: float, u_min: np.ndarray, u_max: np.ndarray,
         rtol: float = RTOL, atol: float = ATOL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From state ``x`` (n,) with input ``u`` (m,) and disturbance ``d`` (r,) over ``Ts``, for
    x' = f(x, u, d): the next state (n,), the input the plant applied (m,), ``u`` clipped to
    ``[u_min, u_max]`` (±inf for no bound), and which channels the clip changed (m,) bool."""
    u, d = np.asarray(u, dtype=float), np.asarray(d, dtype=float)
    applied = np.clip(u, u_min, u_max)
    x_next = hold(lambda x_, u_: f(x_, u_, d), x, applied, (0.0, Ts), rtol, atol)[1][-1]
    return x_next, applied, applied != u
