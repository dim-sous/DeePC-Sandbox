"""Data generation using persistently exciting input signals.

Same PRBS/multisine approach as v1-v3 but with wider excitation
amplitudes and more data to cover a broader operating envelope.
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
    """Generate a pseudo-random binary sequence (PRBS).

    Switches between +amplitude and -amplitude with random hold times
    drawn uniformly from [min_period, 2 * min_period].
    """
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
    """Generate a multi-sine excitation signal.

    Sum of sinusoids at random frequencies with random phases,
    normalized to the requested peak amplitude.
    """
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
    """Collect persistently exciting input-output data from the vehicle.

    The vehicle starts at a steady state (straight-line, constant velocity)
    and is driven with PRBS steering and multi-sine acceleration excitation.
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

    # Generate excitation signals: uniform random for PE guarantee,
    # mixed with PRBS and multisine for richer spectral content
    delta_prbs = generate_prbs(
        T, config.input_amplitude_delta, config.prbs_min_period, rng
    )
    delta_rand = rng.uniform(
        -config.input_amplitude_delta,
        config.input_amplitude_delta,
        size=T,
    )
    delta_exc = 0.5 * delta_prbs + 0.5 * delta_rand

    a_sine = generate_multisine(
        T, config.input_amplitude_a, n_freqs=10, Ts=config.Ts, rng=rng
    )
    a_rand = rng.uniform(-config.input_amplitude_a, config.input_amplitude_a, size=T)
    a_exc = 0.5 * a_sine + 0.5 * a_rand

    u_data = np.zeros((T, config.m))
    y_data = np.zeros((T, config.p))

    for k in range(T):
        u_k = np.array([delta_exc[k], a_exc[k]])
        y_k = sim.step(u_k)
        # Add measurement noise
        y_k = y_k + rng.normal(0, config.noise_std_output, size=config.p)
        u_data[k] = u_k
        y_data[k] = y_k

    return u_data, y_data
