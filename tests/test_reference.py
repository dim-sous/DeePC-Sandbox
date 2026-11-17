"""The safety net: today's code against its frozen numbers.

A branch that means to move no number (a move, a rename, a new layer around the code) proves it
by keeping these green; one that means to move them remakes the files
(``uv run python -m tests.reference.make``) and says so in its PR. The rule is BESS's, written
after its incident B4 (a proof that compared nothing): every array is compared, not a key's
presence.

How it compares is the developer's (2026-11-17, "Bit for bit, per computer"): on the computer
that made the files (``tests/reference/cases.py``, ``computer()``, as the manifest records it),
byte for byte; on another, each finite value of a float array within 1e-9 of the largest finite
|value| that array holds, every non-finite value (inf, -inf, NaN) where it was, strings and
integers equal, and the pass says it compared within tolerance, not bit for bit (a warning and a
recorded property). The old net, made in the cloud, differed on the developer's computer in the
last bit (``.claude/memory/project_state.md``). Known limit: an array whose values are all of
rounding size, such as the closed loop's ‖σ_y‖₂, is held at its own scale, and another
computer's rounding may move it more than that.

What each case runs, and its conditions, are in ``tests/reference/cases.py`` and, as made, in
``tests/reference/manifest.json``.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from tests.reference import cases

MANIFEST = json.loads(cases.MANIFEST.read_text())
#: The relative tolerance off the computer that made the files (the developer, 2026-11-17).
RTOL = 1e-9


def _computers() -> str:
    made, now = MANIFEST["made"]["computer"], cases.computer()
    return "; ".join(f"{k} {made.get(k)} (now {now.get(k)})" for k in made)


def compare(frozen: np.ndarray, now: np.ndarray, exact: bool) -> tuple[str | None, float]:
    """Where ``now`` differs from ``frozen``, in words (None when it does not), and the largest
    relative change |now - frozen| / max|frozen| of a float array (0 otherwise). ``exact``:
    byte for byte; otherwise within the tolerance of the module."""
    if frozen.shape != now.shape or frozen.dtype != now.dtype:
        return f"shape/dtype {frozen.shape} {frozen.dtype} -> {now.shape} {now.dtype}", 0.0
    if frozen.size == 0:
        return None, 0.0
    floating = np.issubdtype(frozen.dtype, np.floating)
    largest = 0.0
    if floating:
        finite = np.isfinite(frozen) & np.isfinite(now)
        scale = float(np.abs(frozen[np.isfinite(frozen)]).max()) if np.isfinite(frozen).any() \
            else 0.0
        if finite.any():
            change = float(np.abs(now[finite] - frozen[finite]).max())
            largest = change / scale if scale else change
    if exact:
        a = np.ascontiguousarray(frozen).reshape(frozen.size, -1).view(np.uint8)
        b = np.ascontiguousarray(now).reshape(now.size, -1).view(np.uint8)
        unequal = (a != b).any(axis=1)
    elif floating:
        # a non-finite value is where it was: an inf bound, a NaN with no answer
        same_kind = (np.isnan(frozen) & np.isnan(now)) | (np.isinf(frozen) & (frozen == now))
        within = finite & (np.abs(np.where(finite, now - frozen, 0.0)) <= RTOL * scale)
        unequal = ~(within | same_kind).reshape(-1)
    else:
        unequal = (frozen != now).reshape(-1)
    if not unequal.any():
        return None, largest
    first = np.unravel_index(int(np.flatnonzero(unequal)[0]), frozen.shape)
    words = (f"{int(unequal.sum())} of {frozen.size} entries differ, first at "
             f"{tuple(map(int, first))}: {frozen[first]!r} -> {now[first]!r}")
    if floating:
        words += f"; largest change {largest:.3g} of the array's largest |value|"
    return words, largest


def test_every_frozen_file_is_a_case_and_every_case_is_frozen() -> None:
    """A case taken out of the registry would otherwise drop out of the net with its file left
    behind, and the suite stay green."""
    frozen = {p.stem for p in cases.REFERENCE_DIR.glob("*.npz")}
    assert frozen == set(cases.CASES), (f"files with no case: {sorted(frozen - set(cases.CASES))}; "
                                        f"cases with no file: {sorted(set(cases.CASES) - frozen)}")
    assert set(MANIFEST["cases"]) == set(cases.CASES)
    assert MANIFEST["left_out"] == cases.LEFT_OUT


@pytest.mark.parametrize("name", cases.CASES)
def test_the_case_runs_under_its_frozen_conditions(name: str) -> None:
    """Every setting the case ran under, as made: a default changed under a case is a change to
    its conditions even where no number moves."""
    assert json.loads(json.dumps(cases.conditions_of(name))) == MANIFEST["cases"][name]


@pytest.mark.parametrize("name", cases.CASES)
def test_the_case_produces_its_frozen_arrays(name: str, record_property) -> None:
    path = cases.REFERENCE_DIR / f"{name}.npz"
    assert path.exists(), f"{name} has no frozen file; make it: uv run python -m tests.reference.make"
    with np.load(path) as stored:
        frozen = {key: stored[key] for key in stored.files}
    now = cases.produce(name)
    assert sorted(now) == sorted(frozen), f"{name}: arrays {sorted(frozen)} -> {sorted(now)}"
    exact = cases.computer() == MANIFEST["made"]["computer"]
    results = {key: compare(frozen[key], now[key], exact) for key in frozen}
    changed = {key: words for key, (words, _) in results.items() if words}
    assert not changed, (f"{name} moved ({'byte for byte' if exact else f'within {RTOL}'}): "
                         + "; ".join(f"{k}: {w}" for k, w in changed.items())
                         + f". Computer: {_computers()}")
    if not exact:
        largest = max(change for _, change in results.values())
        record_property("compared", f"within {RTOL}, not bit for bit")
        warnings.warn(f"{name}: compared within {RTOL}, not bit for bit, on a computer other than "
                      f"the one that made the files (largest change {largest:.3g} of an array's "
                      f"largest |value|). Computer: {_computers()}", stacklevel=1)


def test_off_the_computer_that_made_the_files_a_case_is_compared_in_tolerance_and_says_so(
        monkeypatch) -> None:
    monkeypatch.setattr(cases, "computer", lambda: {**MANIFEST["made"]["computer"],
                                                     "cpu": "another"})
    with pytest.warns(UserWarning, match="compared within 1e-09, not bit for bit"):
        test_the_case_produces_its_frozen_arrays("system-default", lambda *_: None)


def test_the_comparison_holds_bytes_on_one_computer_and_the_tolerance_on_another() -> None:
    frozen = np.array([1.0, -2.0, 0.0, np.nan, 1e-14])
    ulp = frozen.copy()
    ulp[0] = np.nextafter(1.0, 2.0)
    assert compare(frozen, frozen.copy(), exact=True)[0] is None
    assert "1 of 5 entries differ, first at (0,)" in compare(frozen, ulp, exact=True)[0]
    assert compare(frozen, ulp, exact=False)[0] is None
    nudged = frozen.copy()
    nudged[1] *= 1 + 1e-6
    assert "first at (1,)" in compare(frozen, nudged, exact=False)[0]
    assert "shape/dtype" in compare(frozen, frozen[:2], exact=False)[0]
    assert "1 of 2 entries" in compare(np.array(["a", "b"]), np.array(["a", "c"]), exact=False)[0]
    # a non-finite value is held where it was, and does not loosen the rest (code review)
    bound = np.array([-np.inf, -5.0])
    assert "first at (1,)" in compare(bound, np.array([-np.inf, -4.0]), exact=False)[0]
    assert "first at (0,)" in compare(bound, np.array([-1e300, -5.0]), exact=False)[0]
    assert compare(np.zeros(0), np.zeros(0), exact=True)[0] is None
