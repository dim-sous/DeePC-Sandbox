"""The linearisation of a parsed system at an operating point (x*, u*), with no disturbance:
x' ≈ f(x*, u*, 0) + A (x - x*) + B (u - u*), y ≈ h(x*) + C (x - x*), with A = ∂f/∂x, B = ∂f/∂u and
C = ∂h/∂x at the point, exact (SymPy's derivatives, ``Parsed.jacobians``); D = 0, since h names no
input. The point need not be an equilibrium: the residual |f(x*, u*, 0)| says how far it is."""

from __future__ import annotations

import numpy as np

from system.parse import Parsed

__all__ = ["linearize"]


def linearize(parsed: Parsed, x_star: np.ndarray, u_star: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(A, B, C, residual)`` at ``x_star`` (n,), ``u_star`` (m,) and d = 0; the residual is the
    2-norm of f(x*, u*, 0)."""
    x_star, u_star = np.asarray(x_star, dtype=float), np.asarray(u_star, dtype=float)
    no_d = np.zeros(parsed.r)
    A, B, C = parsed.jacobians(x_star, u_star, no_d)
    residual = float(np.linalg.norm(parsed.f(x_star, u_star, no_d)))
    for name, value in (("A", A), ("B", B), ("C", C)):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} at x* = {x_star.tolist()}, u* = {u_star.tolist()} is not "
                             f"finite: {value.tolist()}")
    return A, B, C, residual
