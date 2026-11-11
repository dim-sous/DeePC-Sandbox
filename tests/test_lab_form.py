"""The form, the rail rows and the compare table as plain data (BESS's ``tests/test_lab_form.py``,
adapted): what a control offers and says, which settings a gate shuts, the sweep row, and the
``deepc run`` the form launches -- read off the pieces' own declarations, never restated here."""

from __future__ import annotations

import pytest

from harness import get_component
from lab import state, view
from lab.form import CODE_DEFAULT, Form, as_text, hover, term
from record.config import coerce_value

#: The closed loop's: it applies only while the noise-adaptive λ is on.
_W = "noise_estimation_window"


@pytest.fixture(scope="module")
def piece() -> dict:
    return state.pieces()[0]


@pytest.fixture(scope="module")
def dataset() -> dict:
    return next(p for p in state.pieces() if p["name"] == "dataset")


def _runs(*names: str) -> list[dict]:
    return [{"piece": "closed_loop", "name": n, "config": {}, "provenance": {}, "stale": ""}
            for n in names]


def test_the_lab_opens_on_the_closed_loop(piece) -> None:
    assert piece["name"] == "closed_loop"


def test_a_gate_that_shuts_takes_the_value_behind_it_back_with_the_reason(piece) -> None:
    form = Form(piece)
    assert form.type_or_pick("noise_adaptive", "true") == []
    assert form.type_or_pick(_W, "10") == [] and form.set == {"noise_adaptive": "true", _W: "10"}
    assert form.type_or_pick("noise_adaptive", "false") == [_W]
    assert form.set == {}
    w, gate = form.label(_W), form.label("noise_adaptive")
    off, words = form.state_of(form.by_name[_W])
    assert off and words == f"{w} applies only when {gate} is true; it is false"
    # taken back, and SAID: a typed value that vanishes silently is one the run would ignore
    assert form.note() == (f"0 set — {w} taken back: the run would ignore it", True)


def test_a_gate_is_read_against_the_sweep_rungs_not_the_axis_default(piece) -> None:
    form = Form(piece)
    assert form.sweep("noise_adaptive", "true,false") == []
    assert form.type_or_pick(_W, "10") == []                  # some rungs open it: kept
    off, words = form.state_of(form.by_name[_W])
    assert not off and f"only some sweep values of {form.label('noise_adaptive')} open it" in words
    assert form.note() == (f"1 set — {form.label(_W)} is set, and only some sweep values open it",
                           True)
    assert form.sweep("noise_adaptive", "false") == [_W]       # no rung opens it: taken back
    assert form.state_of(form.by_name[_W])[0]
    form.sweep("noise_adaptive", "true, false, false")
    words, warns = form.note()
    assert warns and "repeats a value — the terminal refuses the run" in words


def test_the_sweep_axis_is_swept_once_and_not_also_set(piece) -> None:
    form = Form(piece)
    form.type_or_pick("lambda_g", "10")
    form.type_or_pick("N", "20")
    assert form.sweep("lambda_g", "1,5") == ["lambda_g"]
    label = form.by_name["lambda_g"]["label"]
    assert form.state_of(form.by_name["lambda_g"]) == (
        True, f"{label} is the sweep axis; its values come from the sweep row")
    assert form.launch_words() == (["N=20"], ["lambda_g=1,5"])
    assert form.sweep("lambda_g", "") == [] and form.swept() == ""
    assert form.launch_words() == (["N=20"], [])
    # the values go as they were typed: what is an acceptable list is the terminal's rule
    form.sweep("lambda_g", " 1 , 5 ")
    assert form.launch_words()[1] == ["lambda_g=1 , 5"]


def test_a_pick_list_offers_the_default_first(piece) -> None:
    form = Form(piece)
    by = form.by_name
    assert [(o["value"], o["words"]) for o in form.options(by["scenario"])] == [
        ("", "default (default)"), ("sinusoidal", "sinusoidal")]
    assert [(o["value"], o["words"]) for o in form.options(by["noise_adaptive"])] == [
        ("", "false (default)"), ("true", "true")]
    assert form.options(by["lambda_g"]) is None


def test_a_default_the_code_resolves_is_said_so_not_left_blank(piece, dataset) -> None:
    """``None`` in the Config is the code's own value, resolved by the run; an empty placeholder
    and a pick worded " (default)" said nothing about it."""
    form = Form(piece)
    by = form.by_name
    assert by["lambda_g"]["code_default"] and form.shown_default(by["lambda_g"]) == CODE_DEFAULT
    assert [(o["value"], o["words"]) for o in form.options(by["reg_norm_g"])] == [
        ("", CODE_DEFAULT), ("L1", "L1"), ("L2", "L2")]
    assert Form(dataset).shown_default(Form(dataset).by_name["seed"]) == "42"


def test_a_saved_run_setting_offers_that_pieces_runs_by_name(piece) -> None:
    """RCWS's ``record_of``: the dataset run is picked from the dataset piece's saved runs, each
    worded by its name with its path as its line; the run it holds stays offered."""
    form = Form(piece)
    f = form.by_name["dataset_run"]
    assert f["record_of"] == "dataset"
    assert form.options(f) == [{"value": "", "words": "default (default)",
                                "line": "results/dataset/default.json"}]
    form.saved = {"dataset": ["results/dataset/noisy.json", "results/dataset/default.json"],
                  "closed_loop": ["results/closed_loop/x.json"]}
    assert [(o["value"], o["words"]) for o in form.options(f)] == [
        ("", "default (default)"), ("results/dataset/noisy.json", "noisy")]
    form.type_or_pick("dataset_run", "elsewhere/gone.json")
    assert [o["words"] for o in form.options(f)] == ["default (default)", "noisy", "gone"]
    assert form.launch_words() == (["dataset_run=elsewhere/gone.json"], [])
    assert "dataset_run" not in dict(form.axes()), "a path cannot name a sweep's records"


def test_a_list_of_numbers_launches_as_the_terminal_reads_it_and_is_no_sweep_axis(piece) -> None:
    """Q and R are lists: a run's own list loads into its box as the JSON ``--set`` reads back,
    and the sweep row does not offer them, because the sweep splits its values on commas."""
    form = Form(piece)
    form.use({"Q_diag": [10.0, 10.0, 1.0]})
    assert form.launch_words() == (["Q_diag=[10.0, 10.0, 1.0]"], [])
    q = get_component("closed_loop").Config.param["Q_diag"]
    assert coerce_value(q, form.set["Q_diag"]) == [10.0, 10.0, 1.0]
    assert {"Q_diag", "R_diag"}.isdisjoint(dict(form.axes()))


def test_a_hover_carries_the_bounds_a_field_declares_as_an_interval(piece) -> None:
    by = Form(piece).by_name
    assert hover(by["sim_duration"]).endswith("[0 … ∞)")
    assert hover({"doc": "d", "low": 0, "high": 100, "low_exclusive": True,
                  "high_exclusive": False}) == "d (0 … 100]", "an included end is a square bracket"
    assert hover(by["scenario"]) == by["scenario"]["doc"]      # no bounds, no interval
    assert term(by["sim_duration"]) == "Duration [s]" and term(by["scenario"]) == "Scenario"


def test_a_name_says_when_run_overwrites_a_record(piece) -> None:
    form = Form(piece)
    assert form.free_name(_runs("closed_loop-1", "closed_loop-2", "other")) == "closed_loop-3"
    assert form.free_name([]) == "closed_loop-1"
    form.name = "other"
    assert form.name_note(_runs("other")) == "other exists — Run overwrites it"
    assert form.name_note(_runs("else")) == ""

    # a sweep writes one record per rung, named as the terminal names them
    held = _runs("closed_loop-1-lambda_g-2", "closed_loop-1-lambda_g-5")
    fresh = Form(piece)
    assert fresh.free_name(held) == "closed_loop-2"
    fresh.sweep("lambda_g", "2,5")
    assert fresh.would_write("closed_loop-1") == ["closed_loop-1-lambda_g-2",
                                                  "closed_loop-1-lambda_g-5"]
    assert fresh.name_note(held) == ""                 # the offered name is free: nothing to warn
    fresh.name = "closed_loop-1"
    assert fresh.name_note(held) == ("closed_loop-1-lambda_g-2, closed_loop-1-lambda_g-5 exist "
                                     "— Run overwrites them")


def test_a_value_is_written_the_way_set_reads_it_back() -> None:
    assert [as_text(v) for v in (None, True, False, 2, 2.0, 0.25, 1e-06, "hold")] == [
        "", "true", "false", "2", "2", "0.25", "1e-06", "hold"]


def test_typing_the_default_leaves_the_field_unset(dataset) -> None:
    form = Form(dataset)
    assert form.type_or_pick("seed", "42") == [] and form.set == {}
    form.type_or_pick("seed", " 7 ")
    assert form.set == {"seed": "7"} and form.effective("seed") == "7"
    assert form.effective("plant") == "coupled_masses"


def test_use_loads_what_a_run_held_away_from_the_defaults_and_clear_empties(piece) -> None:
    form = Form(piece)
    form.type_or_pick("lambda_g", "10")
    form.name, form.sweep_field, form.sweep_values = "x", "N", "10,20"
    n, dropped = form.use({"dataset_run": "results/dataset/noisy.json", "lambda_g": 50.0,
                           "N": None, "scenario": "default", "ancient": 7})
    assert n == 2 and form.set == {"dataset_run": "results/dataset/noisy.json", "lambda_g": "50"}
    assert dropped == ["ancient"]        # named, not silently loaded as if it still were one
    assert form.name == "" and form.swept() == ""
    form.clear()
    assert form.set == {} and form.launch_words() == ([], [])


def test_a_chart_takes_the_first_group_a_run_names_for_it() -> None:
    """A run whose file predates groups names none, and must not blank the chart's group for the
    runs that do (BESS code review, 2026-10-27)."""
    runs = [{"name": "old", "panels": ["p"], "panel_groups": {}},
            {"name": "new", "panels": ["p"], "panel_groups": {"p": "Solver"}}]
    assert [row["group"] for row in view.chart_rows(runs, [])] == ["Solver"]


def test_a_rail_row_is_the_runs_name_and_why_it_cannot_be_read() -> None:
    """RCWS's row: the name alone, a stale run's reason under it; what sets a run apart is the
    compare table's to say."""
    runs = [{"piece": "closed_loop", "name": "a", "config": {"lambda_g": 5.0},
             "provenance": {"date": "2026-10-04T18:02:11+00:00", "git_sha": "0ff4c32abc",
                            "git_dirty": True},
             "stale": ""},
            {"piece": "closed_loop", "name": "c", "config": {}, "provenance": {},
             "stale": "predates the field sigma_y_norm"}]
    assert view.run_rows(runs) == [{"name": "a", "stale": ""},
                                   {"name": "c", "stale": "predates the field sigma_y_norm"}]
    assert view.when(runs[0]) == "2026-10-04 18:02 · 0ff4c32" and view.when(runs[1]) == ""
    assert view.text(None) == "—" and view.text(0.123456) == "0.1235" and view.text(True) == "true"


def _run(name: str, numbers: dict, config: dict, panels=("input-F1", "cost"),
         specs=None) -> dict:
    """A listed record row, as ``state.records`` hands one over."""
    return {"piece": "closed_loop", "name": name, "config": config, "numbers": numbers,
            "panels": list(panels), "number_specs": list(specs if specs is not None else _SPECS),
            "provenance": {}, "stale": "", "mtime": 0.0, "path": f"{name}.json"}


_SPECS = [{"name": "rmse_position_combined", "label": "Position combined RMSE", "unit": "m", "headline": True,
           "doc": "The plant's own metric", "group": "Outputs"},
          {"name": "solve_time_mean_s", "label": "Mean solve time", "unit": "s",
           "headline": False, "doc": "Mean wall-clock time of a solve", "group": "Solver"}]


def test_the_chart_list_is_the_panels_the_shown_runs_declare_in_their_own_order() -> None:
    both = [_run("a", {}, {}), _run("b", {}, {}, panels=("input-F1", "cost", "lambda-g"))]
    rows = view.chart_rows(both, ["cost"])
    assert [(r["name"], r["open"], r["sub"]) for r in rows] == [
        ("input-F1", False, ""), ("cost", True, ""), ("lambda-g", False, "1 of 2 runs")]
    assert view.chart_rows([], []) == []


def test_the_compare_table_puts_the_numbers_first_and_the_settings_by_group(piece) -> None:
    runs = [_run("a", {"rmse_position_combined": 0.00267, "solve_time_mean_s": 3.0},
                 {"lambda_g": 1.0, "sim_duration": 1.0, "scenario": "default"}),
            _run("b", {"rmse_position_combined": 0.00412},
                 {"lambda_g": 5.0, "sim_duration": 1.0, "scenario": "default"})]
    rows = view.compare_rows(runs, piece["fields"], settings="all")

    assert rows[0] == {"kind": "head", "group": "Outputs", "key": "", "term": "Outputs",
                       "hover": "", "values": [], "differs": False}
    numbers = [r for r in rows if r["kind"] == "number"]
    assert [r["key"] for r in numbers] == ["rmse_position_combined", "solve_time_mean_s"], \
        "a row opens its card"
    assert [r["term"] for r in numbers] == ["Position combined RMSE [m]", "Mean solve time [s]"]
    assert numbers[0]["values"] == ["0.00267", "0.00412"] and numbers[0]["differs"]
    assert numbers[0]["hover"] == "The plant's own metric"
    assert numbers[1]["values"] == ["3", "—"], "a number a run does not carry is an em dash"

    heads = [r["term"] for r in rows if r["kind"] == "head"]
    assert heads == ["Outputs", "Solver"] + list(dict.fromkeys(f["group"] for f in piece["fields"]))
    assert [r["kind"] for r in view.compare_rows(runs, piece["fields"])].count("setting") == 0, \
        "the numbers alone until the settings are asked for"
    old = [dict(r, number_specs=[{k: v for k, v in s.items() if k != "group"} for s in _SPECS])
           for r in runs]
    assert view.compare_rows(old, piece["fields"])[0]["term"] == "Numbers", \
        "a file written before groups lists its numbers under one heading"
    by_term = {r["term"]: r for r in rows if r["kind"] == "setting"}
    fields = {f["name"]: f for f in piece["fields"]}
    lam = by_term[term(fields["lambda_g"])]
    assert lam["values"] == ["1", "5"] and lam["differs"]
    duration = by_term[term(fields["sim_duration"])]
    assert "[0 … ∞)" in duration["hover"], "the bounds ride along in the hover"
    scenario = by_term[term(fields["scenario"])]
    assert not scenario["differs"] and scenario["values"] == ["default", "default"]
    assert by_term[term(fields["N"])]["values"] == ["—", "—"], \
        "a setting the run left to the code is an em dash; its resolved value is a constant"


def test_only_the_settings_that_differ_keeps_every_number_and_drops_the_rest(piece) -> None:
    runs = [_run("a", {"rmse_position_combined": 1.0}, {"lambda_g": 1.0, "scenario": "default"}),
            _run("b", {"rmse_position_combined": 2.0}, {"lambda_g": 5.0, "scenario": "default"})]
    rows = view.compare_rows(runs, piece["fields"], settings="differ")
    settings = [r for r in rows if r["kind"] == "setting"]
    assert [r["values"] for r in settings] == [["1", "5"]]
    assert [r["kind"] for r in rows].count("number") == 2, "numbers are what runs are compared on"
    assert view.compare_count(rows, runs, piece["fields"]) == "1 setting · 1 number differ"
    hidden = view.compare_rows(runs, piece["fields"])
    assert view.compare_count(hidden, runs, piece["fields"]) == "1 setting · 1 number differ", \
        "a setting that differs is counted whether or not the table shows it"
    with pytest.raises(ValueError, match="settings is one of"):
        view.compare_rows(runs, piece["fields"], settings="some")


def test_the_count_says_what_differs_and_never_compares_one_run_with_itself(piece) -> None:
    one = [_run("a", {"rmse_position_combined": 1.0}, {})]
    alone = view.compare_rows(one, piece["fields"])
    assert view.compare_count(alone, one, piece["fields"]) == "2 numbers; tick another run to compare"
    pair = [_run("a", {"rmse_position_combined": 1.0}, {"scenario": "default"}),
            _run("b", {"rmse_position_combined": 1.0}, {"scenario": "default"})]
    same = view.compare_rows(pair, piece["fields"])
    assert view.compare_count(same, pair, piece["fields"]) == "2 numbers, none of them differs"
    assert view.varying([_run("a", {}, {"scenario": "default"}),
                         _run("b", {}, {"scenario": "sinusoidal"})], piece["fields"]) == ["scenario"]


def test_a_row_differs_on_the_saved_values_not_on_the_words_they_are_shown_by(piece) -> None:
    """Four digits hide the sixth, and this table is where a refactor's numbers are checked."""
    runs = [_run("a", {"rmse_position_combined": 0.00267124}, {}),
            _run("b", {"rmse_position_combined": 0.00267139}, {})]
    rows = view.compare_rows(runs, piece["fields"])
    row = next(r for r in rows if r["kind"] == "number")
    assert row["values"] == ["0.002671", "0.002671"] and row["differs"]
    assert view.compare_count(rows, runs, piece["fields"]) == "0 settings · 1 number differ"


def test_a_chart_already_open_keeps_a_row_even_when_no_shown_run_draws_it() -> None:
    rows = view.chart_rows([_run("a", {}, {}, panels=("input-F1",))], ["input-F1", "cost"])
    assert [(r["name"], r["open"], r["sub"]) for r in rows] == [
        ("input-F1", True, ""), ("cost", True, "No shown run draws it")]


def test_the_chart_list_is_in_the_groups_the_records_declare() -> None:
    a = dict(_run("a", {}, {}, panels=("input-F1", "cost", "lambda-g")),
             panel_groups={"input-F1": "Inputs", "cost": "Solver", "lambda-g": "Solver"})
    rows = view.chart_rows([a], [])
    assert [(g, [r["name"] for r in rs]) for g, rs in view.grouped(rows)] == [
        ("Inputs", ["input-F1"]), ("Solver", ["cost", "lambda-g"])]
    assert [r["group"] for r in view.chart_rows([_run("old", {}, {})], [])] == ["", ""], \
        "a file written before groups lists ungrouped"


def test_the_run_filter_keeps_the_runs_holding_every_word_anywhere() -> None:
    runs = [_run(n, {}, {}) for n in ("sweep-lambda_g-1", "sweep-lambda_g-50", "default", "p100")]
    assert [r["name"] for r in view.matching(runs, "lambda 50")] == ["sweep-lambda_g-50"]
    assert [r["name"] for r in view.matching(runs, "  ")] == [r["name"] for r in runs]
    assert view.matching(runs, "lambda nope") == []
