"""Data generation using persistently exciting input signals.

Collects error-based outputs [e_lat, e_heading, e_v] relative to a
straight-line nominal at v_ref.  Multiple episodes at different initial
speeds and headings ensure output diversity.
"""

import numpy as np

from control.config import DeePCConfig
from plants.bicycle_model import BicycleModel, compute_path_errors


def generate_prbs(
    length: int,
    amplitude: float,
    min_period: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a pseudo-random binary sequence (PRBS)."""
    signal = np.empty(length)
    idx = 0
    value = amplitude if rng.random() < 0.5 else -amplitude

    while idx < length:
        hold = rng.integers(min_period, 2 * min_period + 1)
        signal[idx : min(idx + hold, length)] = value
        idx += hold
        value = -value

    return signal


def generate_multisine(
    length: int,
    amplitude: float,
    n_freqs: int,
    Ts: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a multi-sine excitation signal."""
    t = np.arange(length) * Ts
    nyquist = 0.5 / Ts
    freqs = rng.uniform(0.01 * nyquist, 0.4 * nyquist, size=n_freqs)
    phases = rng.uniform(0, 2 * np.pi, size=n_freqs)

    signal = np.zeros(length)
    for f, phi in zip(freqs, phases):
        signal += np.sin(2 * np.pi * f * t + phi)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal *= amplitude / peak

    return signal


def collect_data(
    config: DeePCConfig,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect persistently exciting input-output data.

    Outputs are [e_lat, e_heading, e_v] relative to a straight-line
    nominal trajectory at v_ref heading=0.  Multiple episodes at
    different initial speeds and headings ensure output diversity.
    """
    rng = np.random.default_rng(seed)

    speed_offsets = [0.0, -0.4, 0.4, -0.7, 0.7]
    n_episodes = len(speed_offsets)
    T_per = config.T_data // n_episodes
    remainder = config.T_data - T_per * n_episodes

    u_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    # Stabilizing gains for baseline controller during data collection.
    # These keep errors bounded near zero; excitation is added on top.
    K_lat = 0.3     # steering correction per meter of lateral error
    K_head = 1.0    # steering correction per radian of heading error
    K_v = 0.5       # acceleration correction per m/s of velocity error

    for i, dv in enumerate(speed_offsets):
        T_ep = T_per + (1 if i < remainder else 0)
        v_init = max(config.v_ref + dv * config.v_ref, 0.5)
        heading_init = rng.uniform(-0.3, 0.3)

        sim = BicycleModel(
            Ts=config.Ts,
            L_wheelbase=config.L_wheelbase,
            delta_max=config.delta_max,
            a_max=config.a_max,
            a_min=config.a_min,
            initial_state=np.array([0.0, 0.0, heading_init, v_init]),
        )

        # Excitation signals
        delta_prbs = generate_prbs(
            T_ep, config.input_amplitude_delta, config.prbs_min_period, rng
        )
        delta_rand = rng.uniform(
            -config.input_amplitude_delta,
            config.input_amplitude_delta,
            size=T_ep,
        )
        delta_exc = 0.5 * delta_prbs + 0.5 * delta_rand

        a_sine = generate_multisine(
            T_ep, config.input_amplitude_a, n_freqs=10, Ts=config.Ts, rng=rng
        )
        a_rand = rng.uniform(
            -config.input_amplitude_a, config.input_amplitude_a, size=T_ep
        )
        a_exc = 0.5 * a_sine + 0.5 * a_rand

        u_ep = np.zeros((T_ep, config.m))
        y_ep = np.zeros((T_ep, config.p))

        # Nominal straight-line reference: heading=0, v=v_ref
        nom_x = 0.0

        for k in range(T_ep):
            nom_x += config.v_ref * config.Ts
            errors = compute_path_errors(
                sim.state, nom_x, 0.0, 0.0, config.v_ref
            )

            # Stabilizing baseline + excitation
            u_stab = np.array([
                -K_lat * errors[0] - K_head * errors[1],
                -K_v * errors[2],
            ])
            u_k = u_stab + np.array([delta_exc[k], a_exc[k]])

            sim.step(u_k)
            # Re-compute errors after stepping
            errors = compute_path_errors(
                sim.state, nom_x, 0.0, 0.0, config.v_ref
            )
            errors += rng.normal(0, config.noise_std_output, size=config.p)

            u_ep[k] = u_k
            y_ep[k] = errors

        u_parts.append(u_ep)
        y_parts.append(y_ep)

    return np.vstack(u_parts), np.vstack(y_parts)
