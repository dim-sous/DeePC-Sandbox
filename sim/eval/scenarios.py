"""Reference trajectory generators for DeePC simulation."""

from __future__ import annotations

import numpy as np

from control.config import DeePCConfig


def generate_reference_trajectory(config: DeePCConfig) -> np.ndarray:
    """Generate a sinusoidal reference trajectory (default scenario)."""
    total = config.Tini + config.sim_steps + config.N
    y_ref = np.zeros((total, config.p))
    for k in range(total):
        t = k * config.Ts
        y_ref[k, 0] = config.v_ref * t
        y_ref[k, 1] = config.ref_amplitude * np.sin(
            2 * np.pi * config.ref_frequency * t
        )
        y_ref[k, 2] = config.v_ref
    return y_ref


def generate_hard_reference(config: DeePCConfig) -> np.ndarray:
    """Generate a driving-course trajectory with multiple maneuver phases.

    Phases:
        1. Straight + accelerate      5->8 m/s
        2. Slalom (tight zigzag)       8 m/s
        3. Decelerate into sharp turn  8->3 m/s, ~90 deg right
        4. Tight circle                4 m/s, radius ~10 m
        5. Straighten + accelerate     4->7 m/s
        6. Double lane change          7 m/s (elk test)
        7. Cruise (settle)             7 m/s
    """
    dt = config.Ts
    total = config.Tini + config.sim_steps + config.N

    v_profile = np.zeros(total)
    hdot_profile = np.zeros(total)

    p1 = 50
    p2 = 130
    p3 = 180
    p4 = 260
    p5 = 300
    p6 = 350

    for k in range(total):
        if k < p1:
            frac = k / p1
            v_profile[k] = 5.0 + 3.0 * frac
            hdot_profile[k] = 0.0
        elif k < p2:
            v_profile[k] = 8.0
            t_phase = (k - p1) * dt
            hdot_profile[k] = 1.8 * np.sin(2 * np.pi * 0.5 * t_phase)
        elif k < p3:
            frac = (k - p2) / (p3 - p2)
            v_profile[k] = 8.0 - 5.0 * frac
            turn_center = 0.5
            turn_width = 0.3
            gauss = np.exp(-((frac - turn_center) ** 2) / (2 * turn_width ** 2))
            hdot_profile[k] = -3.0 * gauss
        elif k < p4:
            v_profile[k] = 4.0
            hdot_profile[k] = v_profile[k] / 10.0
        elif k < p5:
            frac = (k - p4) / (p5 - p4)
            v_profile[k] = 4.0 + 3.0 * frac
            hdot_profile[k] = 0.4 * (1.0 - frac)
        elif k < p6:
            v_profile[k] = 7.0
            t_phase = (k - p5) * dt
            period = (p6 - p5) * dt
            hdot_profile[k] = 1.5 * np.sin(2 * np.pi * (2.0 / period) * t_phase)
        else:
            v_profile[k] = 7.0
            hdot_profile[k] = 0.0

    heading = np.zeros(total)
    x = np.zeros(total)
    y = np.zeros(total)

    for k in range(1, total):
        heading[k] = heading[k - 1] + hdot_profile[k - 1] * dt
        x[k] = x[k - 1] + v_profile[k - 1] * np.cos(heading[k - 1]) * dt
        y[k] = y[k - 1] + v_profile[k - 1] * np.sin(heading[k - 1]) * dt

    y_ref = np.zeros((total, config.p))
    y_ref[:, 0] = x
    y_ref[:, 1] = y
    y_ref[:, 2] = v_profile

    return y_ref


def generate_circle_reference(config: DeePCConfig, radius: float = 15.0) -> np.ndarray:
    """Generate a circular trajectory at constant speed."""
    dt = config.Ts
    total = config.Tini + config.sim_steps + config.N
    v = config.v_ref
    omega = v / radius

    heading = np.zeros(total)
    x = np.zeros(total)
    y = np.zeros(total)

    for k in range(1, total):
        heading[k] = heading[k - 1] + omega * dt
        x[k] = x[k - 1] + v * np.cos(heading[k - 1]) * dt
        y[k] = y[k - 1] + v * np.sin(heading[k - 1]) * dt

    y_ref = np.zeros((total, config.p))
    y_ref[:, 0] = x
    y_ref[:, 1] = y
    y_ref[:, 2] = v
    return y_ref


def generate_square_reference(config: DeePCConfig, side: float = 20.0) -> np.ndarray:
    """Generate a square trajectory: straight segments with 90-degree turns."""
    dt = config.Ts
    total = config.Tini + config.sim_steps + config.N
    v = config.v_ref

    side_time = side / v
    side_steps = int(side_time / dt)
    turn_radius = 3.0
    turn_arc = (np.pi / 2) * turn_radius
    turn_steps = max(int(turn_arc / (v * dt)), 5)
    turn_omega = (np.pi / 2) / (turn_steps * dt)

    heading = np.zeros(total)
    x = np.zeros(total)
    y = np.zeros(total)
    v_profile = np.full(total, v)

    leg_steps = side_steps + turn_steps
    for k in range(1, total):
        pos_in_leg = k % leg_steps
        if pos_in_leg < side_steps:
            hdot = 0.0
        else:
            hdot = turn_omega

        heading[k] = heading[k - 1] + hdot * dt
        x[k] = x[k - 1] + v_profile[k - 1] * np.cos(heading[k - 1]) * dt
        y[k] = y[k - 1] + v_profile[k - 1] * np.sin(heading[k - 1]) * dt

    y_ref = np.zeros((total, config.p))
    y_ref[:, 0] = x
    y_ref[:, 1] = y
    y_ref[:, 2] = v_profile
    return y_ref


def generate_zigzag_reference(config: DeePCConfig, amplitude: float = 5.0, seg_length: float = 15.0) -> np.ndarray:
    """Generate a zigzag trajectory: alternating diagonal segments."""
    dt = config.Ts
    total = config.Tini + config.sim_steps + config.N
    v = config.v_ref

    heading_angle = np.arctan2(amplitude, seg_length)
    seg_dist = np.sqrt(seg_length**2 + amplitude**2)
    seg_steps = int(seg_dist / (v * dt))
    seg_steps = max(seg_steps, 5)

    heading = np.zeros(total)
    x = np.zeros(total)
    y = np.zeros(total)

    trans_steps = 5
    target_heading = heading_angle

    for k in range(1, total):
        seg_idx = k // seg_steps
        pos_in_seg = k % seg_steps

        if seg_idx % 2 == 0:
            target_heading = heading_angle
        else:
            target_heading = -heading_angle

        if pos_in_seg < trans_steps:
            frac = pos_in_seg / trans_steps
            blend = 0.5 * (1 - np.cos(np.pi * frac))
            prev_target = -heading_angle if seg_idx % 2 == 0 else heading_angle
            if seg_idx == 0:
                prev_target = 0.0
            heading[k] = prev_target + blend * (target_heading - prev_target)
        else:
            heading[k] = target_heading

        x[k] = x[k - 1] + v * np.cos(heading[k]) * dt
        y[k] = y[k - 1] + v * np.sin(heading[k]) * dt

    y_ref = np.zeros((total, config.p))
    y_ref[:, 0] = x
    y_ref[:, 1] = y
    y_ref[:, 2] = v
    return y_ref


def get_reference(scenario: str, config: DeePCConfig) -> np.ndarray:
    """Dispatch to the appropriate trajectory generator."""
    generators = {
        "default": generate_reference_trajectory,
        "hard": generate_hard_reference,
        "circle": generate_circle_reference,
        "square": generate_square_reference,
        "zigzag": generate_zigzag_reference,
    }
    return generators[scenario](config)
