"""The safety net's cases: what today's code produces at stated conditions, frozen so that a
change meant to move no number can show it moved none (CLAUDE.md, "a refactor proves itself").

Every case runs the repository's own functions as ``run.py`` calls them -- ``collect_data``,
``get_reference``, ``run_simulation``, ``run.main`` -- at seed ``SEED``, every setting at the
plant's default except those the case names. Four kinds:

- ``<plant>-data``: the training data, at the plant's defaults;
- ``<plant>-reference-<scenario>``: each scenario's reference path, at the plant's defaults;
- a closed loop, named below: training data, reference and the DeePC loop, over its duration;
- ``run-py-<plant>``: ``run.py`` end to end, its report written to a temporary folder.

Solve times are wall clock and are not kept. The bugs recorded in ``docs/project/failures.md``
are frozen too: a branch that fixes one moves these files on purpose (``make.py``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import sys
from pathlib import Path

import numpy as np

import run
from control.config import DeePCConfig, build_deepc_config
from plants.base import PlantBase
from sim.data_generation import collect_data
from sim.scenarios import get_reference
from sim.simulation import FeatureFlags, run_simulation

REFERENCE_DIR = Path(__file__).resolve().parent
MANIFEST = REFERENCE_DIR / "manifest.json"
ROOT = REFERENCE_DIR.parent.parent
#: The code the frozen numbers come from; ``make.py`` records a fingerprint of these files.
CODE = ("control", "plants", "sim", "run.py")
#: The seed of every case's training data (``run.py``'s own default).
SEED = 42
#: The closed loops: plant, scenario, duration [s], the feature flags ``run.py`` sets, and the
#: config fields set off the plant's defaults besides ``sim_duration``. The durations are off
#: the plants' defaults (bicycle 60 s, coupled masses 30 s) to keep the suite short, and open.
#: The coupled masses' default steps hold zero for their first 5 s (``_generate_step_sequence``),
#: so its cases run the sinusoid, or the steps past 5 s.
CLOSED_LOOPS: dict[str, tuple[str, str, float, dict, dict]] = {
    "bicycle-default": ("bicycle", "default", 5.0, {}, {}),
    # hankel_warmup_steps 20, off its default 100, so the window slides within the run
    "bicycle-online-hankel": ("bicycle", "default", 5.0, {"online_hankel": True},
                              {"hankel_warmup_steps": 20}),
    "bicycle-noise-adaptive": ("bicycle", "default", 5.0, {"noise_adaptive": True}, {}),
    "coupled_masses-sinusoidal": ("coupled_masses", "sinusoidal", 3.0, {}, {}),
    "coupled_masses-default-no-constraints": ("coupled_masses", "default", 8.0,
                                              {"no_constraints": True}, {}),
}
#: ``run.py`` end to end, one per plant: the bicycle draws its own figures, the coupled masses
#: the generic ones.
RUN_PY: dict[str, tuple[str, ...]] = {
    "run-py-bicycle": ("--plant", "bicycle", "--sim-duration", "3"),
    "run-py-coupled_masses": ("--plant", "coupled_masses", "--scenario", "sinusoidal",
                              "--no-constraints", "--sim-duration", "3"),
}

def _plant(name: str) -> PlantBase:
    return run.PLANT_REGISTRY[name]()


def _open_loop_cases() -> dict[str, tuple[str, str | None]]:
    """``{case: (plant, scenario or None)}``, read off the registry: a plant or a scenario
    added tomorrow is a case tomorrow, refused by the test until its file is made."""
    out: dict[str, tuple[str, str | None]] = {}
    for plant_name in run.PLANT_REGISTRY:
        out[f"{plant_name}-data"] = (plant_name, None)
        for scenario in _plant(plant_name).get_scenarios():
            out[f"{plant_name}-reference-{scenario}"] = (plant_name, scenario)
    return out


OPEN_LOOPS = _open_loop_cases()
#: Every case whose frozen file is ``<case>.npz``.
CASES: tuple[str, ...] = tuple(OPEN_LOOPS) + tuple(CLOSED_LOOPS)


def config_of(name: str) -> DeePCConfig:
    """The DeePCConfig the case runs under, built as ``run.py`` builds it."""
    if name in OPEN_LOOPS:
        return build_deepc_config(_plant(OPEN_LOOPS[name][0]))
    plant_name, _, duration, _, overrides = CLOSED_LOOPS[name]
    return build_deepc_config(_plant(plant_name), sim_duration=duration, **overrides)


def conditions_of(name: str) -> dict:
    """Everything the case ran under, stated in full: the plant, the scenario, the seed, the
    feature flags and every DeePCConfig field."""
    if name in OPEN_LOOPS:
        plant_name, scenario = OPEN_LOOPS[name]
        flags: dict = {}
    else:
        plant_name, scenario, _, flags, _ = CLOSED_LOOPS[name]
    return {"plant": plant_name, "scenario": scenario, "seed": SEED,
            "flags": dataclasses.asdict(FeatureFlags(**flags)),
            "config": dataclasses.asdict(config_of(name))}


def _floats(name: str, key: str, values: list) -> np.ndarray:
    """A per-step list as a float array; a step that holds no value is refused, not saved as NaN."""
    missing = [k for k, v in enumerate(values) if v is None]
    if missing:
        raise ValueError(f"{name}: {key} holds no value at steps {missing[:10]}")
    return np.asarray(values, dtype=float)


def produce(name: str) -> dict[str, np.ndarray]:
    """What today's code produces for case ``name``, as the arrays its frozen file holds."""
    plant_name = OPEN_LOOPS[name][0] if name in OPEN_LOOPS else CLOSED_LOOPS[name][0]
    plant, cfg = _plant(plant_name), config_of(name)
    with contextlib.redirect_stdout(io.StringIO()):          # the pipeline prints progress
        if name in OPEN_LOOPS:
            scenario = OPEN_LOOPS[name][1]
            if scenario is None:
                u_data, y_data = collect_data(plant, cfg, seed=SEED)
                return {"u_data": u_data, "y_data": y_data}
            return {"path": get_reference(plant, scenario, cfg)}
        _, scenario, _, flags, _ = CLOSED_LOOPS[name]
        u_data, y_data = collect_data(plant, cfg, seed=SEED)
        path = get_reference(plant, scenario, cfg)
        res = run_simulation(plant, cfg, FeatureFlags(**flags), u_data, y_data, path)
    out = {key: np.asarray(res[key]) for key in
           ("times", "u_history", "y_history", "ref_path", "statuses")}
    for key in ("costs", "sigma_norms", "lambda_g_history"):
        out[key] = _floats(name, key, res[key])
    # sigma_out_norm is None at every step while lambda_out is 0, which no case here sets off
    if any(v is not None for v in res["sigma_out_norms"]):
        out["sigma_out_norms"] = _floats(name, "sigma_out_norms", res["sigma_out_norms"])
    for key in ("pos_history", "ref_pos_history"):
        if key in res:
            out[key] = np.asarray(res[key])
    return out


@contextlib.contextmanager
def _patched(obj, attr: str, value):
    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old)


def run_py_conditions(name: str) -> dict:
    """Everything ``run.py`` runs under for case ``name``, as ``run.main`` builds it: the
    arguments, every parsed option (its defaults included) and every DeePCConfig field."""
    with _patched(sys, "argv", ["run.py", *RUN_PY[name]]):
        args = run.parse_args()
    config = run.build_config(args, _plant(args.plant))
    return {"argv": ["run.py", *RUN_PY[name]], "options": vars(args),
            "config": dataclasses.asdict(config)}


def frozen_names() -> tuple[set[str], set[str]]:
    """The cases the reference folder holds files for: ``(*.npz stems, run-py *.json stems)``."""
    return ({p.stem for p in REFERENCE_DIR.glob("*.npz")},
            {p.stem for p in REFERENCE_DIR.glob("run-py-*.json")})


def run_py_metrics(name: str, out_dir: Path) -> dict:
    """Run ``run.py`` end to end with case ``name``'s arguments, its report and metrics written
    under ``out_dir``; return the metrics file without its wall-clock numbers
    (``*solve_time_s``). A missing report or metrics file is refused."""
    args = RUN_PY[name]
    with (_patched(run, "RESULTS_DIR", Path(out_dir)),
          _patched(sys, "argv", ["run.py", *args]),
          contextlib.redirect_stdout(io.StringIO())):
        run.main()
    written = sorted(Path(out_dir).rglob("*"))
    reports = [p for p in written if p.suffix == ".html"]
    metrics = [p for p in written if p.name.endswith("_metrics.json")]
    if len(reports) != 1 or len(metrics) != 1:
        raise FileNotFoundError(f"run.py {' '.join(args)} wrote {[str(p) for p in written]}; "
                                f"expected one report and one metrics file")
    return {k: v for k, v in json.loads(metrics[0].read_text()).items()
            if not k.endswith("solve_time_s")}
