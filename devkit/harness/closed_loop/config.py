"""The closed-loop piece's settings: the saved dataset run it runs on, the scenario and its length,
the controller's settings, and the three features ``run.py`` switches by flag. The step T_s is the
dataset run's (its plant's), not a setting here (D1).

A setting left at ``None`` takes the code's own default -- the plant's
(``get_default_config_overrides``), else ``DeePCConfig``'s -- exactly as ``run.py`` leaves a flag
it was not given; the run saves what it resolved (``constants["deepc_config"]``). The settings of
the two optional features apply only while their feature is on (``needs``). The labels and help
lines follow the developer's pattern for the words a Lab user reads (CLAUDE.md;
``docs/project/lab-words.md``); the two features' groups show closed until opened (``folded``).
"""

from __future__ import annotations

import param

from plants import PLANT_REGISTRY
from record.config import Settings

DATA, SCENARIO, CONTROLLER, FEATURES = "Data", "Scenario", "Controller", "Features"
NOISE_ADAPTIVE, ONLINE_HANKEL = "Noise-adaptive", "Online Hankel"
GROUPS: tuple[str, ...] = (DATA, SCENARIO, CONTROLLER, FEATURES, NOISE_ADAPTIVE, ONLINE_HANKEL)

#: The settings handed to ``build_deepc_config`` by name (``None``: the code's own default).
DEEPC_SETTINGS: tuple[str, ...] = (
    "sim_duration", "Tini", "N", "Q_diag", "R_diag", "lambda_g", "lambda_y", "lambda_out",
    "reg_norm_g", "reg_norm_sigma_y", "noise_estimation_window", "baseline_noise_std",
    "max_lambda_scaling", "constraint_tightening_factor", "hankel_window_size",
    "hankel_warmup_steps")
#: The settings handed to ``FeatureFlags`` by name.
FEATURE_SETTINGS: tuple[str, ...] = ("noise_adaptive", "online_hankel", "no_constraints")


def _scenarios() -> list[str]:
    """Every scenario any registered plant offers; the run refuses one its plant lacks."""
    names: list[str] = []
    for plant in PLANT_REGISTRY.values():
        names += [s for s in plant().get_scenarios() if s not in names]
    return names


def _own(cls, label: str, doc: str, group: str, unit: str = "-", **kw) -> param.Parameter:
    """A setting whose ``None`` default is the code's own value, in ``unit``."""
    meta = {"group": group, "unit": unit, **({"folded": True} if "needs" in kw else {})}
    if "needs" in kw:
        meta["needs"] = kw.pop("needs")
    return cls(default=None, allow_None=True, label=label,
               doc=f"{doc}; none: the code's own default", metadata=meta, **kw)


_ON = {"in": [True]}


class Config(Settings):
    """The closed-loop piece's settings; see the module."""

    dataset_run = param.String(
        "results/dataset/default.json", label="Dataset run",
        doc="The saved dataset run the controller's Hankel matrices are built from; run again "
            "first and checked against the file; its plant is the plant",
        metadata={"group": DATA, "record_of": "dataset"})
    scenario = param.Selector(
        default="default", objects=_scenarios(), label="Scenario",
        doc="The reference the plant is asked to follow; one the plant offers",
        metadata={"group": SCENARIO})
    sim_duration = _own(param.Number, "Duration", "Closed-loop duration in seconds",
                        SCENARIO, unit="s", bounds=(0.0, None))
    Tini = _own(param.Integer, "T_ini", "Past window length in samples", CONTROLLER, unit="samples")
    N = _own(param.Integer, "N", "Prediction horizon in samples", CONTROLLER, unit="samples")
    Q_diag = _own(param.List, "Q", "Output weights, one per output", CONTROLLER,
                  item_type=float)
    R_diag = _own(param.List, "R", "Input weights, one per input", CONTROLLER,
                  item_type=float)
    lambda_g = _own(param.Number, "λ_g", "Weight of the regulariser on g", CONTROLLER)
    lambda_y = _own(param.Number, "λ_y", "Weight of the slack σ_y on the past outputs", CONTROLLER)
    lambda_out = _own(param.Number, "λ_out",
                      "Weight of the slack on the output bounds; 0: no output bound", CONTROLLER)
    reg_norm_g = _own(param.Selector, "Norm on g", "Regulariser norm on g (L2: the squared 2-norm, D14)",
                      CONTROLLER, objects=["L1", "L2"])
    reg_norm_sigma_y = _own(param.Selector, "Norm on σ_y",
                            "Norm on the past-output slack (L2: the squared 2-norm)", CONTROLLER,
                            objects=["L1", "L2"])
    noise_adaptive = param.Boolean(
        False, label="Noise-adaptive λ",
        doc="Rescale the regularisation each step from the prediction residuals (D16)",
        metadata={"group": FEATURES})
    online_hankel = param.Boolean(
        False, label="Online Hankel",
        doc="Slide the Hankel window with closed-loop data (D5, D17)",
        metadata={"group": FEATURES})
    no_constraints = param.Boolean(
        False, label="No constraints",
        doc="Drop every input, rate and output bound from the controller; the plant still clips",
        metadata={"group": FEATURES})
    noise_estimation_window = _own(param.Integer, "W",
                                   "Residuals kept by the noise estimator", NOISE_ADAPTIVE,
                                   needs={"noise_adaptive": _ON})
    baseline_noise_std = _own(param.Number, "σ_0",
                              "Noise level the scaling divides by", NOISE_ADAPTIVE,
                              needs={"noise_adaptive": _ON})
    max_lambda_scaling = _own(param.Number, "k_max", "Largest scaling of λ_g and λ_y",
                              NOISE_ADAPTIVE, needs={"noise_adaptive": _ON})
    constraint_tightening_factor = _own(param.Number, "c",
                                        "Output-bound tightening per estimated std (D11)",
                                        NOISE_ADAPTIVE, needs={"noise_adaptive": _ON})
    hankel_window_size = _own(param.Integer, "Hankel columns",
                              "Columns the online window holds; 0: the offline count",
                              ONLINE_HANKEL, needs={"online_hankel": _ON})
    hankel_warmup_steps = _own(param.Integer, "Warm-up steps",
                               "Steps before the window starts sliding", ONLINE_HANKEL,
                               needs={"online_hankel": _ON})
