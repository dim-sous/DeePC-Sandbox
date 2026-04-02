"""Data generation using persistently exciting input signals.

v3: adds chirp steering and phased speed sweeps for better coverage
of the operating envelope, while maintaining PE guarantees.
"""

import numpy as np

from config.parameters import DeePCConfig
from plants.bicycle_model import BicycleModel


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


def generate_chirp(
    length: int,
    amplitude: float,
    f_start: float,
    f_end: float,
    Ts: float,
) -> np.ndarray:
    """Generate a linear chirp signal sweeping from f_start to f_end."""
    t = np.arange(length) * Ts
    T_total = length * Ts
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * T_total))
    return amplitude * np.sin(phase)


def collect_data(
    config: DeePCConfig,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect persistently exciting input-output data.

    Uses a three-phase excitation strategy:
      Phase 1 (40%): PRBS + random (classic PE, aggressive)
      Phase 2 (30%): Chirp steering + speed ramp (sweep operating regimes)
      Phase 3 (30%): Multisine + gentle random (fill spectral gaps)

    This covers a wider operating envelope than pure PRBS while
    maintaining persistent excitation.
    """
    rng = np.random.default_rng(seed)
    sim = BicycleModel(
        Ts=config.Ts,
        L_wheelbase=config.L_wheelbase,
        delta_max=config.delta_max,
        a_max=config.a_max,
        a_min=config.a_min,
        initial_state=np.array([0.0, 0.0, 0.0, config.v_ref]),
    )

    T = config.T_data
    t1 = int(0.4 * T)  # end of phase 1
    t2 = int(0.7 * T)  # end of phase 2

    # ── Phase 1: PRBS + random (aggressive PE) ─────────────────
    delta_prbs = generate_prbs(
        t1, config.input_amplitude_delta, config.prbs_min_period, rng
    )
    delta_rand = rng.uniform(
        -config.input_amplitude_delta, config.input_amplitude_delta, size=t1
    )
    delta_p1 = 0.5 * delta_prbs + 0.5 * delta_rand

    a_rand = rng.uniform(-config.input_amplitude_a, config.input_amplitude_a, size=t1)
    a_p1 = a_rand

    # ── Phase 2: chirp steering + speed sweep ──────────────────
    n2 = t2 - t1
    nyquist = 0.5 / config.Ts
    delta_p2 = generate_chirp(
        n2, config.input_amplitude_delta * 0.8,
        f_start=0.02 * nyquist, f_end=0.3 * nyquist, Ts=config.Ts,
    )
    # Speed ramp: accelerate then decelerate to explore different speeds
    a_ramp = np.zeros(n2)
    third = n2 // 3
    a_ramp[:third] = config.input_amplitude_a * 0.6  # accelerate
    a_ramp[third:2*third] = 0.0  # cruise at higher speed
    a_ramp[2*third:] = -config.input_amplitude_a * 0.6  # decelerate
    # Add small noise for PE
    a_p2 = a_ramp + rng.uniform(-0.2 * config.input_amplitude_a,
                                 0.2 * config.input_amplitude_a, size=n2)

    # ── Phase 3: multisine + gentle random (spectral fill) ─────
    n3 = T - t2
    delta_sine = generate_multisine(
        n3, config.input_amplitude_delta * 0.6, n_freqs=8, Ts=config.Ts, rng=rng
    )
    delta_gentle = rng.uniform(
        -0.3 * config.input_amplitude_delta,
        0.3 * config.input_amplitude_delta, size=n3
    )
    delta_p3 = delta_sine + delta_gentle

    a_sine = generate_multisine(
        n3, config.input_amplitude_a * 0.5, n_freqs=6, Ts=config.Ts, rng=rng
    )
    a_p3 = a_sine + rng.uniform(-0.3 * config.input_amplitude_a,
                                 0.3 * config.input_amplitude_a, size=n3)

    # ── Concatenate and simulate ───────────────────────────────
    delta_exc = np.concatenate([delta_p1, delta_p2, delta_p3])
    a_exc = np.concatenate([a_p1, a_p2, a_p3])

    u_data = np.zeros((T, config.m))
    y_data = np.zeros((T, config.p))

    for k in range(T):
        u_k = np.array([delta_exc[k], a_exc[k]])
        y_k = sim.step(u_k)
        y_k = y_k + rng.normal(0, config.noise_std_output, size=config.p)
        u_data[k] = u_k
        y_data[k] = y_k

    return u_data, y_data
