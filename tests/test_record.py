"""The record layer: a saved run loads back exactly, and what it cannot hold is refused in words."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import param
import pytest

from record import Number, Provenance, Record, Series, read_header, stamp
from record.config import Settings, from_dict, with_overrides


class _Config(Settings):
    gain = param.Number(1.0, bounds=(0.0, None), metadata={"group": "G"})
    mode = param.Selector(default="a", objects=["a", "b"], metadata={"group": "G"})
    extra = param.Integer(default=None, allow_None=True,
                          metadata={"group": "G", "needs": {"mode": {"in": ["b"]}}})


@dataclasses.dataclass(frozen=True)
class _Record(Record):
    name: str
    config: _Config
    provenance: Provenance | None
    values: np.ndarray
    labels: list
    numbers: dict

    component = "probe"
    produced_by = "the test"

    def series(self):
        return (Series(name="values", x=np.arange(len(self.values)), x_label="i",
                       y_label="v", traces=(("v", self.values),)),)

    def number_specs(self):
        return (Number("total", "total"),)


def _rec(**kw) -> _Record:
    base = dict(name="r", config=_Config(gain=2.5), provenance=stamp(0.1),
                values=np.array([0.1, 1 / 3, np.nan, np.inf, -np.inf, 1e-300]),
                labels=["optimal", "optimal_inaccurate"], numbers={"total": 3.0})
    return _Record(**{**base, **kw})


def test_a_saved_record_loads_back_bit_for_bit(tmp_path) -> None:
    rec = _rec()
    back = _Record.load(rec.save(tmp_path / "r.json"))
    assert back.values.tobytes() == rec.values.tobytes()
    assert back.labels == rec.labels and back.numbers == rec.numbers
    assert back.config.gain == 2.5 and back.provenance == rec.provenance
    assert read_header(tmp_path / "r.json")["component"] == "probe"


@pytest.mark.parametrize("numbers, words", [({"total": math.nan}, "not finite"),
                                            ({"total": 1.0, "other": 2.0}, "carry no words")])
def test_a_number_that_is_not_finite_or_not_declared_is_refused(tmp_path, numbers, words) -> None:
    with pytest.raises(ValueError, match=words):
        _rec(numbers=numbers).save(tmp_path / "r.json")


@pytest.mark.parametrize("edit, words", [(lambda raw: raw.update(stranger=1), "unknown fields"),
                                         (lambda raw: raw.pop("labels"), "missing"),
                                         (lambda raw: raw.update(component="other"), "not 'probe'")])
def test_a_file_that_is_not_this_record_is_refused(tmp_path, edit, words) -> None:
    path = _rec().save(tmp_path / "r.json")
    raw = json.loads(path.read_text())
    edit(raw)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=words):
        _Record.load(path)


def test_a_setting_is_refused_in_words() -> None:
    with pytest.raises(ValueError, match="unknown setting"):
        from_dict(_Config, {"gian": 1.0})
    with pytest.raises(ValueError, match="applies only when"):
        _Config(extra=3)
    assert with_overrides(_Config(), [("mode", "b"), ("extra", "3")]).extra == 3   # a gate opened in the same call
    assert with_overrides(_Config(mode="b", extra=3), [("extra", "none")]).extra is None
