"""``run(config) -> Record``: DeePC in closed loop on a saved open-loop run. The open-loop run is
read through ``harness.open_loop.open_loop_of`` (run again and checked); its inputs and outputs
are the controller's data, and its plant is the loop's plant: the same system, T_s, actuator
bounds, σ, σ_d and integrator tolerances, sampled by the same two functions (``system.measure``,
``system.step``), so the output the loop gives the controller and the one the data holds come
from one formula (D4, D10, D12). The controller (``deepc.controller.DeePCController``) is used as
it is, its maths unchanged (the developer, 2026-11-17). No printing, no saving, no drawing.

At every sample k from x_0: the output y_k = h(x_k) + n_k is measured; for k < T_ini the input is
zero (the developer, 2026-11-17, as the old loop's warm-up), after that DeePC solves for u_k from
the last T_ini inputs and outputs and the reference over the horizon; the plant clips u_k to its
bounds, holds it with d_k over T_s and integrates. A solve that leaves no solution stops the run
(the developer, 2026-11-17).
"""

from __future__ import annotations

import time

import numpy as np

from benchlab import stamp
from deepc.config import DeePCConfig
from deepc.controller import DeePCController
from harness.open_loop import open_loop_of
from harness.system import system_of
from system import measure, parse_signal, step

from .config import Config
from .record import Record


def _sized(name: str, values: list, size: int, what: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.shape != (size,):
        raise ValueError(f"{name} holds {out.size} value{'' if out.size == 1 else 's'}; the "
                         f"system has {size} {what}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} = {out.tolist()} is not finite")
    return out


def _weights(name: str, values: list, size: int, what: str) -> np.ndarray:
    out = _sized(name, values, size, what)
    # the controller takes the square root of each weight (``deepc.controller``): a negative one
    # would be NaN there
    if np.any(out < 0):
        raise ValueError(f"{name} = {out.tolist()}: a weight may not be negative")
    return out


#: The fields of ``DeePCConfig`` the controller reads (``src/deepc/controller.py``,
#: ``online_hankel.py``), recorded as it read them; the rest of that config is bench settings this
#: piece does not use, and is not recorded as if it had.
READ_BY_CONTROLLER: tuple[str, ...] = (
    "m", "p", "Tini", "N", "Q_diag", "R_diag", "lambda_g", "lambda_y", "reg_norm_g",
    "reg_norm_sigma_y", "lambda_out", "hankel_window_size", "solver", "solver_verbose")


def solve_counts(statuses: list[str], solvers: list[str]) -> dict[str, int]:
    """Every solve counted by its outcome: ``solves_<status>`` per status, matched exactly (D2),
    and ``solves_by_<solver>`` per solver that answered (D3)."""
    out = {f"solves_{s}": statuses.count(s) for s in dict.fromkeys(statuses)}
    out.update({f"solves_by_{s}": solvers.count(s) for s in dict.fromkeys(solvers)})
    return out


def run(cfg: Config) -> Record:
    started = time.perf_counter()
    data = open_loop_of(cfg.open_loop_run)
    oc = data.config
    plant = system_of(oc.system_run)          # the system the open-loop run was checked against
    n, m, p, r = plant.n, plant.m, plant.p, plant.r
    x0 = data.x[0] if cfg.x0 is None else _sized("x0", cfg.x0, n, "states")
    Q = _weights("Q", cfg.Q, p, "outputs")
    R = _weights("R", cfg.R, m, "inputs")
    if len(cfg.r) != p:
        raise ValueError(f"r holds {len(cfg.r)} row{'' if len(cfg.r) == 1 else 's'}; the system "
                         f"has {p} outputs, one reference row each")
    r_read, reference = parse_signal(cfg.r)
    for name in ("duration", "lambda_g", "lambda_y"):     # the Config's bounds let inf through
        if not np.isfinite(getattr(cfg, name)):
            raise ValueError(f"{name} = {getattr(cfg, name)} is not finite")
    Ts = oc.Ts
    # solves after the warm-up, the old loop's count int(duration / T_s), but a duration that is a
    # multiple of T_s as typed gives that many: 0.35 / 0.05 reads 6.999999999999999 in floats
    # (code review, 2026-11-17)
    quotient = cfg.duration / Ts
    steps = round(quotient) if np.isclose(quotient, round(quotient), rtol=1e-12, atol=0) \
        else int(quotient)
    if steps < 1:
        raise ValueError(f"duration = {cfg.duration} is shorter than one step T_s = {Ts} of the "
                         f"open-loop run: no solve would run")
    K = cfg.Tini + steps
    u_min, u_max = np.asarray(oc.u_min, dtype=float), np.asarray(oc.u_max, dtype=float)
    sigma_d = np.asarray(oc.sigma_d, dtype=float)
    deepc_cfg = DeePCConfig(m=m, p=p, Tini=cfg.Tini, N=cfg.N, Q_diag=Q.tolist(),
                            R_diag=R.tolist(), lambda_g=cfg.lambda_g, lambda_y=cfg.lambda_y,
                            reg_norm_g=cfg.reg_norm_g, reg_norm_sigma_y=cfg.reg_norm_sigma_y)
    # the open-loop run holds the actuator bounds only: no rate bound and no output bound reaches
    # the controller (the developer's 3c, docs/project/design.md)
    controller = DeePCController(deepc_cfg, data.u, data.y, u_min, u_max,
                                 np.full(m, -np.inf), np.full(m, np.inf),
                                 np.full(p, -np.inf), np.full(p, np.inf))
    rng = np.random.default_rng(cfg.seed)
    noise = oc.sigma * rng.standard_normal(size=(K, p))
    # drawn after the noise, so a system with no disturbance keeps it; the same model as the
    # open-loop run's (its MISMATCH on d_k, ``harness/open_loop/run.py``)
    d = sigma_d * rng.standard_normal(size=(K, r))
    # the reference over every sample and the horizon past the last one
    ref = reference(np.arange(K + cfg.N - 1) * Ts)
    if not np.all(np.isfinite(ref)):
        k = int(np.argmax(~np.all(np.isfinite(ref), axis=1)))
        raise ValueError(f"r at t = {k * Ts} is {ref[k].tolist()}: not finite")
    x, u, y = np.empty((K, n)), np.empty((K, m)), np.empty((K, p))
    clean = np.empty((K, p))                                  # h(x_k), the output before noise
    statuses, solvers, cost, sigma_norm, solve_time = [], [], [], [], []
    clipped = np.zeros(m, dtype=int)
    state = np.asarray(x0, dtype=float)
    for k in range(K):
        if not np.all(np.isfinite(state)):
            raise ValueError(f"the state at sample {k} (t = {k * Ts}) is {state.tolist()}: not "
                             f"finite")
        x[k] = state
        clean[k], y[k] = measure(plant.h, state, noise[k])
        if k < cfg.Tini:
            wanted = np.zeros(m)
        else:
            # the past window holds samples k - T_ini … k - 1 of both u and y: a Hankel column
            # pairs u_j with y_j = h(x_j) (the open-loop run's timing), and u_k is not yet chosen
            ticked = time.perf_counter()
            wanted, info = controller.solve(u[k - cfg.Tini:k], y[k - cfg.Tini:k],
                                            ref[k:k + cfg.N], u_prev=u[k - 1])
            solve_time.append(time.perf_counter() - ticked)
            if info["no_solution"]:
                raise RuntimeError(f"the solve at sample {k} (t = {k * Ts}) left no solution: "
                                   f"status {info['status']!r} from {info['solver']}; the run "
                                   f"stops here")
            statuses.append(str(info["status"]))
            solvers.append(str(info["solver"]))
            cost.append(float(info["cost"]))
            sigma_norm.append(float(info["sigma_y_norm"]))
        state, u[k], changed = step(plant.f, state, wanted, d[k], Ts, u_min, u_max, oc.rtol,
                                    oc.atol)
        clipped += changed
    solve_time = np.asarray(solve_time)
    sigma_norm = np.asarray(sigma_norm)
    numbers = {f"y{i + 1}_rmse": float(np.sqrt(np.mean((clean[:, i] - ref[:K, i]) ** 2)))
               for i in range(p)}
    numbers.update(solve_counts(statuses, solvers))
    numbers.update({"solve_time_mean": float(solve_time.mean()),
                    "solve_time_std": float(solve_time.std()),
                    "solve_time_max": float(solve_time.max()),
                    "sigma_y_norm_mean": float(sigma_norm.mean()),
                    "sigma_y_norm_max": float(sigma_norm.max()),
                    "clipped": int(clipped.sum())})
    prov = data.provenance
    return Record(
        name="default", config=cfg, provenance=stamp(time.perf_counter() - started),
        constants={"open_loop_run": {"path": cfg.open_loop_run, "name": data.name,
                                     "date": prov.date if prov else "",
                                     "git_sha": prov.git_sha if prov else ""},
                   # the plant as the open-loop run holds it: the loop's plant is that run's
                   "f_read": list(plant.f_read), "h_read": list(plant.h_read),
                   "theta": list(plant.theta), "n": n, "m": m, "p": p, "r": r, "Ts": Ts,
                   "u_min": u_min.tolist(), "u_max": u_max.tolist(), "sigma": oc.sigma,
                   "sigma_d": sigma_d.tolist(), "rtol": oc.rtol, "atol": oc.atol,
                   "data_seed": oc.seed, "x0": np.asarray(x0, dtype=float).tolist(),
                   "r_read": list(r_read), "steps": steps,
                   "excitation_rank": controller.excitation_rank,
                   "excitation_rank_required": controller.excitation_rank_required,
                   "deepc_config": {k: getattr(deepc_cfg, k) for k in READ_BY_CONTROLLER},
                   "clipped_per_input": clipped.tolist()},
        t=np.arange(K) * Ts, u=u, y=y, y_true=clean, r=ref[:K], x=x, d=d,
        statuses=statuses, solvers=solvers, cost=np.asarray(cost), sigma_y_norm=sigma_norm,
        solve_time=solve_time, numbers=numbers)
