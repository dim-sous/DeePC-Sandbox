"""``run(config) -> Record``: simulate a saved system run open loop. From x0, at every sample k:
the output y_k = h(x_k) + n_k is measured, as Coulson et al. [1, eq. (1)] y(t) = Cx(t) + Du(t)
(METHODS.md §11); an input u_k is drawn i.i.d. uniform on its excitation range, as [1, §VI-B], and
a disturbance d_k Gaussian; the plant clips u_k to its actuator bounds, holds both over T_s and
integrates (``system.step``). No baseline controller (the developer, 2026-11-11). No printing, no
saving, no drawing.
"""

from __future__ import annotations

import time

import numpy as np

from deepc.regularization import check_persistent_excitation
from harness.system import system_of
from benchlab.record import stamp
from system import step

from .config import Config
from .record import Record


def _sized(name: str, values: list, size: int, what: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.shape != (size,):
        raise ValueError(f"{name} holds {out.size} value{'' if out.size == 1 else 's'}; the "
                         f"system has {size} {what}")
    if np.any(np.isnan(out)):
        raise ValueError(f"{name} = {out.tolist()} holds NaN")
    return out


def _finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} = {values.tolist()} is not finite")


def _ordered(low_name: str, low: np.ndarray, high_name: str, high: np.ndarray) -> None:
    if np.any(low > high):
        j = int(np.argmax(low > high))
        raise ValueError(f"u{j + 1}: {low_name} {low[j]} is above {high_name} {high[j]}")


def pe_order(u: np.ndarray) -> int:
    """The largest depth L at which the input is persistently exciting of order L, its Hankel
    matrix H_L(u) of full row rank L·m ([1, Def. 4.4]), by the controller's own check
    (``deepc.regularization.check_persistent_excitation``, its rank tolerance), bisected over
    L <= (T + 1)/(m + 1), where H_L(u) has at least as many columns as rows; 0 when H_1(u) is
    rank deficient. Full row rank at L implies it at L - 1, so the bisection finds the largest."""
    T, m = u.shape

    def full(L: int) -> bool:
        rank, needed = check_persistent_excitation(u, L, m)
        return rank == needed

    lo, hi = 0, (T + 1) // (m + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if full(mid) else (lo, mid - 1)
    return lo


def run(cfg: Config) -> Record:
    started = time.perf_counter()
    plant = system_of(cfg.system_run)
    x0 = _sized("x0", cfg.x0, plant.n, "states")
    u_min, u_max = (_sized(k, getattr(cfg, k), plant.m, "inputs") for k in ("u_min", "u_max"))
    e_min, e_max = (_sized(k, getattr(cfg, k), plant.m, "inputs")
                    for k in ("u_exc_min", "u_exc_max"))
    _finite("x0", x0)
    for name in ("Ts", "sigma", "rtol", "atol"):         # the Config's bounds let inf through
        _finite(name, np.array([getattr(cfg, name)]))
    # a bound with no limit is -inf below or inf above; inf below or -inf above holds u nowhere
    if np.any(u_min == np.inf) or np.any(u_max == -np.inf):
        raise ValueError(f"u_min = {u_min.tolist()}, u_max = {u_max.tolist()}: u_min may be -inf "
                         f"(no lower limit) and u_max inf, not the other way round")
    _ordered("u_min", u_min, "u_max", u_max)
    _ordered("u_exc_min", e_min, "u_exc_max", e_max)
    sigma_d = _sized("sigma_d", cfg.sigma_d, plant.r, "disturbances")
    _finite("sigma_d", sigma_d)
    if np.any(sigma_d < 0):
        raise ValueError(f"sigma_d = {sigma_d.tolist()}: a standard deviation is not negative")
    for name, ends in (("u_exc_min", e_min), ("u_exc_max", e_max)):
        if not np.all(np.isfinite(ends)):
            raise ValueError(f"{name} = {ends.tolist()}: an excitation is drawn from a finite "
                             f"range")
    rng = np.random.default_rng(cfg.seed)
    drawn = rng.uniform(e_min, e_max, size=(cfg.T, plant.m))
    noise = cfg.sigma * rng.standard_normal(size=(cfg.T, plant.p))
    # drawn after the excitation and the noise, so a system with no disturbance keeps both.
    # MISMATCH: d_k is Gaussian with a fixed σ_d per sample, held over T_s (the developer's model,
    # 2026-11-11); it is not a sampled continuous-time white noise of fixed intensity, whose
    # per-sample variance would scale with 1/T_s, so the same σ_d pushes differently at another T_s.
    d = sigma_d * rng.standard_normal(size=(cfg.T, plant.r))
    x, u, y, clipped = np.empty((cfg.T, plant.n)), np.empty((cfg.T, plant.m)), \
        np.empty((cfg.T, plant.p)), np.zeros(plant.m, dtype=int)
    clean = np.empty((cfg.T, plant.p))                       # h(x_k), the output before noise
    step_times = np.empty(cfg.T)
    state = x0
    for k in range(cfg.T):
        if not np.all(np.isfinite(state)):
            raise ValueError(f"the state at sample {k} (t = {k * cfg.Ts}) is {state.tolist()}: "
                             f"not finite")
        x[k], clean[k] = state, plant.h(state)
        y[k] = clean[k] + noise[k]
        ticked = time.perf_counter()
        state, u[k], changed = step(plant.f, state, drawn[k], d[k], cfg.Ts, u_min, u_max,
                                    cfg.rtol, cfg.atol)
        step_times[k] = time.perf_counter() - ticked
        clipped += changed
    numbers = {"pe_order": pe_order(u), "clipped": int(clipped.sum()),
               "step_time_mean": float(step_times.mean()), "step_time_std": float(step_times.std()),
               "step_time_max": float(step_times.max())}
    if cfg.sigma > 0:                 # no noise, no ratio: a number with no answer is left out
        numbers.update({f"y{j + 1}_snr": float(np.var(clean[:, j]) / cfg.sigma ** 2)
                        for j in range(plant.p)})
    return Record(
        name="default", config=cfg, provenance=stamp(time.perf_counter() - started),
        # the system as read, not only its path: the file may be written again (D1)
        constants={"system_run": cfg.system_run, "f_read": list(plant.f_read),
                   "h_read": list(plant.h_read), "theta": list(plant.theta), "n": plant.n,
                   "m": plant.m, "p": plant.p, "q": plant.q, "r": plant.r,
                   "clipped_per_input": clipped.tolist()},
        t=np.arange(cfg.T) * cfg.Ts, u=u, y=y, x=x, d=d, numbers=numbers)
