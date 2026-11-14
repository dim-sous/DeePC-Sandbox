"""``run(config) -> Record``: read the typed rows with their θ values and check them
(``system.parse``), then analyse the system at the requested point (x*, u*) for the pair
u_j -> y_i: its exact linearisation (``system.linearize``) and, by python-control on that linear
model, its poles, the pair's frequency response, step, impulse and ramp responses, DC gain and
stability margins, over the ranges python-control chooses. No printing, no saving, no drawing.
And ``system_of(path)``: a saved system run, read again and checked, for the pieces that run on one.
"""

from __future__ import annotations

import math
import os
import time

import control
import numpy as np

from benchlab.record import stamp
from system import Parsed, linearize, parse

from .config import Config
from .record import Record


def _point(name: str, values: list, size: int, what: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.shape != (size,):
        raise ValueError(f"{name} holds {out.size} value{'' if out.size == 1 else 's'}; the "
                         f"system has {size} {what}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} = {out.tolist()} is not finite")
    return out


def _index(name: str, value: int, size: int, letter: str) -> int:
    if not 1 <= value <= size:
        raise ValueError(f"{name} = {value}: the system has {letter}1…{letter}{size}")
    return value - 1


def _read(cfg: Config) -> tuple[Record, Parsed]:
    """The run's record and the system it read, from one parse."""
    started = time.perf_counter()
    parsed = parse(cfg.f, cfg.h, cfg.theta)
    x_star = _point("x_star", [0.0] * parsed.n if cfg.x_star is None else cfg.x_star, parsed.n,
                    "states")
    u_star = _point("u_star", [0.0] * parsed.m if cfg.u_star is None else cfg.u_star, parsed.m,
                    "inputs")
    j = _index("input", cfg.input, parsed.m, "u")
    i = _index("output", cfg.output, parsed.p, "y")
    A, B, C, residual = linearize(parsed, x_star, u_star)
    # G ≡ 0 exactly when every Markov parameter C_i A^k B_j, k < n, is 0 (Cayley-Hamilton)
    markov = [float((C[[i], :] @ np.linalg.matrix_power(A, k) @ B[:, [j]])[0, 0])
              for k in range(parsed.n)]
    if not any(markov):
        raise ValueError(f"u{j + 1} does not reach y{i + 1} at (x*, u*): G ≡ 0 (C_i A^k B_j = 0 "
                         f"for k < {parsed.n}); pick another pair")
    pair = control.ss(A, B[:, [j]], C[[i], :], np.zeros((1, 1)))
    frequency = control.frequency_response(pair)              # python-control's own ω range
    omega = np.asarray(frequency.omega, dtype=float)
    magnitude = np.asarray(frequency.magnitude, dtype=float).reshape(-1)
    phase_deg = np.degrees(np.unwrap(np.asarray(frequency.phase, dtype=float).reshape(-1)))
    stepped = control.step_response(pair)                      # python-control's own time span
    t = np.asarray(stepped.time, dtype=float)
    impulse = control.impulse_response(pair, T=t)
    ramp = control.forced_response(pair, T=t, U=t)
    # the poles of A itself, all its modes, not the pair's (which a reduction could trim)
    _, zeta, poles = control.ss(A, B, C, np.zeros((parsed.p, parsed.m))).damp()
    gain_margin, phase_margin, *_ = control.stability_margins(pair)
    dc_gain = control.dcgain(pair)
    for name, values in (("omega", omega), ("magnitude", magnitude), ("phase", phase_deg),
                         ("step", stepped.outputs), ("impulse", impulse.outputs),
                         ("ramp", ramp.outputs)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"python-control's {name} for u{j + 1} -> y{i + 1} at (x*, u*) is "
                             f"not finite")
    numbers = {"n": parsed.n, "m": parsed.m, "p": parsed.p, "q": parsed.q, "r": parsed.r,
               "unstable_poles": int(np.sum(np.real(poles) > 0)),
               "max_pole_re": float(np.max(np.real(poles))), "residual": residual}
    left_out = {}
    # a damping ratio is undefined at a pole at the origin (python-control gives NaN there)
    defined = np.asarray(zeta, dtype=float)[np.isfinite(zeta)]
    min_damping = float(defined.min()) if defined.size else float("nan")
    # a number with no finite answer is left out, never saved as inf or NaN; the reason is kept
    for name, value in (("min_damping", min_damping), ("dc_gain", dc_gain),
                        ("gain_margin", gain_margin), ("phase_margin", phase_margin)):
        value = float(np.real(value))
        if math.isfinite(value):
            numbers[name] = value
        elif name == "min_damping":
            left_out[name] = "every pole is at the origin, where a damping ratio is undefined"
        else:
            left_out[name] = f"python-control gives {value}"
    record = Record(
        name="default", config=cfg, provenance=stamp(time.perf_counter() - started),
        constants={"f_read": list(parsed.f_read), "h_read": list(parsed.h_read),
                   "A": A.tolist(), "B": B.tolist(), "C": C.tolist(),
                   "poles_re": np.real(poles).tolist(), "poles_im": np.imag(poles).tolist(),
                   "x_star": x_star.tolist(), "u_star": u_star.tolist(),
                   "pair": f"u{j + 1} -> y{i + 1}",
                   "omega_range": [float(omega[0]), float(omega[-1])],
                   "t_end": float(t[-1]), "left_out": left_out},
        omega=omega, magnitude=magnitude, phase_deg=phase_deg, t=t,
        step=np.asarray(stepped.outputs, dtype=float).reshape(-1),
        impulse=np.asarray(impulse.outputs, dtype=float).reshape(-1),
        ramp=np.asarray(ramp.outputs, dtype=float).reshape(-1), numbers=numbers)
    return record, parsed


def run(cfg: Config) -> Record:
    return _read(cfg)[0]


def system_of(path: str) -> Parsed:
    """The system the saved run at ``path`` defines, checked: its settings are read again and
    must give the rows as read and the sizes the file holds -- a file written by code this
    checkout no longer holds is refused, with the command that writes it again (as
    ``harness.dataset.dataset_of``). Only the rows are read again, and their reading and sizes
    compared: the analysis at (x*, u*) is the System tab's, and a piece that simulates the system
    neither needs it nor should fail on it."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no saved system run at {path}; the system piece writes one: "
                                f"{Record.produced_by} [--name NAME]")
    saved = Record.load(path)
    # the rows alone, not the analysis: what a piece running on the system needs is the system
    parsed = parse(saved.config.f, saved.config.h, saved.config.theta)
    again = {"f_read": list(parsed.f_read), "h_read": list(parsed.h_read), "n": parsed.n,
             "m": parsed.m, "p": parsed.p, "q": parsed.q, "r": parsed.r}
    held = {**{k: saved.constants.get(k) for k in ("f_read", "h_read")},
            **{k: saved.numbers.get(k) for k in "nmpqr"}}
    differ = [k for k in again if held[k] != again[k]]
    if differ:
        raise ValueError(f"the system run at {path} does not read again to what it holds "
                         f"({', '.join(differ)} differ): code this checkout no longer holds wrote "
                         f"it; run it again ({Record.produced_by})")
    return parsed
