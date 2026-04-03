"""Reference path generators for DeePC simulation.

Each generator returns a (total, 4) array of path waypoints:
    [x, y, heading, v]
at each time step.  The DeePC controller operates on errors relative
to these waypoints, not on the waypoints directly.
"""

from __future__ import annotations

import numpy as np

from control.config import DeePCConfig


def generate_reference_path(config: DeePCConfig) -> np.ndarray:
    """Generate a sinusoidal reference path (default scenario).

    The vehicle moves forward at v_ref with sinusoidal oscillations in
    both x and y (x at frequency f, y at 2f).
    """
    total = config.Tini + config.sim_steps + config.N
    dt = config.Ts
    A = config.ref_amplitude
    f = config.ref_frequency

    path = np.zeros((total, 4))  # [x, y, heading, v]
    for k in range(total):
        t = k * dt
        path[k, 0] = config.v_ref * t + A * np.sin(2 * np.pi * f * t)
        path[k, 1] = A * np.sin(2 * np.pi * 2 * f * t)

    # Heading from finite differences of position
    for k in range(total - 1):
        dx = path[k + 1, 0] - path[k, 0]
        dy = path[k + 1, 1] - path[k, 1]
        path[k, 2] = np.arctan2(dy, dx)
        path[k, 3] = np.sqrt(dx ** 2 + dy ** 2) / dt
    path[-1, 2] = path[-2, 2]
    path[-1, 3] = path[-2, 3]

    return path


def generate_lissajous_path(config: DeePCConfig) -> np.ndarray:
    """Generate a Lissajous figure-8 reference path.

    x = A * sin(f * t),  y = A * sin(2f * t).
    A closed curve — the vehicle must manage its own speed.
    """
    total = config.Tini + config.sim_steps + config.N
    dt = config.Ts
    A = config.ref_amplitude
    f = config.ref_frequency

    path = np.zeros((total, 4))
    for k in range(total):
        t = k * dt
        path[k, 0] = A * np.sin(2 * np.pi * f * t)
        path[k, 1] = A * np.sin(2 * np.pi * 2 * f * t)

    for k in range(total - 1):
        dx = path[k + 1, 0] - path[k, 0]
        dy = path[k + 1, 1] - path[k, 1]
        path[k, 2] = np.arctan2(dy, dx)
        path[k, 3] = np.sqrt(dx ** 2 + dy ** 2) / dt
    path[-1, 2] = path[-2, 2]
    path[-1, 3] = path[-2, 3]

    return path


def get_reference(scenario: str, config: DeePCConfig) -> np.ndarray:
    """Dispatch to the appropriate path generator."""
    generators = {
        "default": generate_reference_path,
        "lissajous": generate_lissajous_path,
    }
    return generators[scenario](config)
