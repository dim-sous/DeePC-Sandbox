"""The one integrator: x' = f(x, u) with u held constant over a span (zero-order hold), by
``scipy.integrate.solve_ivp``, method RK45 -- the Dormand-Prince 5(4) pair the SciPy
documentation names (J. R. Dormand, P. J. Prince, J. Comput. Appl. Math. 6(1), 1980; not yet
checked against the paper).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["RTOL", "ATOL", "hold"]

#: The default tolerances, the developer's decision 1a (docs/project/design.md, 2026-11-11); the
#: error they give against an exact answer is checked in tests/test_system.py.
RTOL, ATOL = 1e-9, 1e-12


def hold(f: Callable[[np.ndarray, np.ndarray], np.ndarray], x0: np.ndarray, u: np.ndarray,
         span: tuple[float, float], rtol: float = RTOL,
         atol: float = ATOL) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ``f`` from ``x0`` over ``span`` with ``u`` held constant. Returns the solver's
    own time points ``(K,)`` and the states at them ``(K, n)``, the first being ``x0``. A failed
    solve raises with the solver's message."""
    u = np.asarray(u, dtype=float)
    sol = solve_ivp(lambda _t, x: f(x, u), span, np.asarray(x0, dtype=float), method="RK45",
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"RK45 failed over t in {span}: {sol.message}")
    return sol.t, sol.y.T
