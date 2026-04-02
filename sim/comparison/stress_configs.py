"""Common stress test configurations shared across all gate runs.

Each function returns a dict of parameter overrides to pass to DeePCConfig.
Using a shared definition ensures the stress test scenarios are identical
regardless of feature flags.
"""

# Common base overrides applied to every stress test config.
_COMMON = dict(
    T_data=200,
    noise_std_output=0.01,
    input_amplitude_delta=0.3,
    input_amplitude_a=1.0,
    sim_steps=150,
)


def base(**overrides) -> dict:
    """Return merged common + per-test overrides."""
    cfg = dict(_COMMON)
    cfg.update(overrides)
    return cfg


def high_noise() -> dict:
    return base(noise_std_output=0.1)


def aggressive_reference() -> dict:
    return base(ref_amplitude=10.0, ref_frequency=0.1)


def reduced_data() -> dict:
    return base(T_data=50, sim_steps=50)


def tight_constraints() -> dict:
    return base(delta_max=0.1, a_max=1.0, a_min=-1.0)


def nonlinear_regime() -> dict:
    return base(v_ref=10.0, ref_amplitude=8.0)


def step_reference() -> dict:
    return base(ref_amplitude=0.0)


def disturbance_rejection() -> dict:
    return base()


def long_horizon() -> dict:
    return base(sim_steps=200, T_data=200)


def rate_constrained_aggressive() -> dict:
    return base(ref_amplitude=10.0, ref_frequency=0.1)
