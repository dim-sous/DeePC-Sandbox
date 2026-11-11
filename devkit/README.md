# devkit/ — the test bench

Runs the controller (`src/`) on the simulated plants (`tools/`), records each run, draws the
records, and shows them in the Lab. May import everything; nothing in `src/` or `tools/`
imports it (`tests/test_roots.py`). Who imports whom inside it is held by
`tests/test_devkit_layers.py`.

| package | holds |
|---|---|
| `sim/` | the simulation code the pieces drive: data collection (`collect_data`, `episode_lengths`), the scenarios (`get_reference`), the closed loop (`run_simulation`), the `DeePCConfig` built from a plant (`build_deepc_config`), the solves counted by outcome (`solve_counts`), the metrics `tune.py` scores by (`metrics.compute_metrics`, moved from the retired `run.py`) and a plant's input bounds as records keep them (`bounds.input_bounds_of`) |
| `record/` | the settings layer (`record.config.Settings` on `param`: every setting with a group, refused in words when wrong, `needs` for a setting that applies only while another holds a named value, `None` for "the code's own default", `record_of` for a setting naming another piece's saved run) and the record layer (`Record`, saved as one JSON and loaded back bit for bit; `Series` and `Number`, the charts and numbers a record declares; `Provenance`, what produced it) |
| `harness/` | the registry of pieces (`NAMES`, `get_component`: reading it imports no piece) and one package per piece, each exposing `Config`, `run(config) -> Record` and `Record`. `dataset`: training data from a plant at its step T_s, as `collect_data` makes it, episodes stacked end to end. `closed_loop`: DeePC on a saved dataset run (re-run first and checked against its file, `harness.dataset.dataset_of`), following one of its plant's scenarios; every solve's status, cost, slack and time kept, solves counted by status (D2), stepping at its dataset run's T_s (`harness.dataset.plant_of`, D1) |
| `figure/` | one panel spec (`render.Panel`) built by `chart` from what records declare and drawn three ways: a PNG still (`render`), the Lab's live twin (`twin.figure_of`, a plotly figure as a plain dict) and the numbers on it (`render.table_of`, a CSV beside the still); BESS's figure layer without its chi-square band. A panel may give each trace its own abscissa (`trace_x`: a path, y against x) and declare equal scales (`equal_axes`). `chart.chart_of` draws a panel over one or several runs of one piece -- on one plot (the colour carrying the run or the quantity, the other a shade of that hue and a dash; a trace every run holds identically drawn once and named so: "a, b — Reference (identical)"), a row each, or pooled by the mean / RMS the panel declares (the dataset's channels, over runs that differ in their seed alone) -- with log axes, dots, the sample index or a span of the bottom axis; it refuses a panel some of the runs do not declare, which drawn without them would leave them out unsaid. `chart.sweep_of` draws a summary number against a setting, every run a point, coloured by the settings that tell the runs apart but a replicate. What it cannot draw it refuses in words (`pool_refusal`, `log_refusal`). Axis labels with units, a legend only when there is more than one label, no title. Every save goes through `style.save_figure`: the PNG and a JSON sidecar (commit, date, command, runs) |
| `terminal/` | `deepc run <piece>`: build the Config, apply each `--set`, run once per `--sweep` combination, save one record per run. `deepc draw <record>...`: the panels the records declare as PNGs beside them, or `--number N --vs SETTING`; every way a draw is shaped declared once (`DRAW_SHAPING`), which the Lab's Save builds its command from |
| `lab/` | the Lab, BESS's ported: one local page on Panel, one screen. The piece's runs on the left -- a list to tick, filter by words, tick all or none of, and delete from after it asks; the ticked runs compared column by column, the numbers in the groups their record declares, the settings when asked (None, Differing, All) and the constants the run resolved (`deepc_config · lambda_g`); the folders of stills a Save wrote. Their cards on the right: a panel the records declare, opened from the chart list in its groups, or a number clicked in the compare table, drawn against the setting that varies. Every card carries the same controls, declared once (`lab/shaping.py`): X, Log x, Log y; Colour by; Overlay, Split, Mean, RMS (on a number card All, Mean, RMS); Dots or Lines; Chart or Table -- a control the card cannot use greyed with the figure layer's reason on its hover; Save writes the still exactly as the card shows it. A draggable split; the form over the settings in a drawer (a gate greyed with its reason, the dataset run picked from the saved ones by name, the two features' groups folded until opened, a setting the code resolves shown as "Code default", a sweep row, Run and Stop); the terminal's own output along the bottom. It launches `deepc run` and `deepc draw` as subprocesses (the line is printed so it can be typed), reads declarations, record files and a shown record through its class, and computes nothing itself (`tests/test_lab.py` holds the import allowlist and drives the page against a piece that raises if run; `tests/test_lab_browser.py` drives the served page in headless Chromium: what is pasted or typed with Tab is what the run uses, every chart's points are its record's numbers, nothing runs off the page or overprints at 1000 and 1440 px) |

Commands, from the repository folder:

```
uv run deepc run dataset                                   # bicycle, its own defaults -> results/dataset/default.json
uv run deepc run dataset --set plant=coupled_masses --name masses
uv run deepc run closed_loop                               # DeePC on results/dataset/default.json -> results/closed_loop/default.json
uv run deepc run closed_loop --set dataset_run=results/dataset/masses.json --set scenario=sinusoidal --name sine
uv run deepc run closed_loop --set sim_duration=10 --sweep lambda_g=1,10,100 --name lg
                                                           # one record per value: results/closed_loop/lg-lambda_g-<v>.json
uv run deepc run closed_loop --set online_hankel=true --set hankel_warmup_steps=20
                                                           # a feature's own settings apply only while it is on
uv run deepc run closed_loop my-setup.json                 # a saved settings file is the way to keep a setup
uv run deepc draw results/closed_loop/default.json         # every panel it declares -> results/closed_loop/default/<panel>.png
uv run deepc draw results/closed_loop/lg-*.json --panel output-e-lateral
                                                           # the runs on one plot -> results/closed_loop/<runs>/output-e-lateral.png
uv run deepc draw results/closed_loop/lg-*.json --number rmse_position --vs lambda_g
                                                           # a number against the swept setting, every run a point
uv run deepc draw results/closed_loop/lg-*.json --panel cost --runs rows --table
                                                           # a row each, and the numbers on it as CSV
uv run lab --no-open                                       # the Lab on http://127.0.0.1:8765: run a piece, look at what it wrote
```

A setting left unset takes the code's own default, the plant's where it has one, and the record
saves what it resolved (`constants`). A record holds the settings the run used, the arrays its
charts draw, summary numbers each a reduction of one of them, and its provenance (commit,
dirty flag, date, duration, library versions). `results/` is git-ignored: a run never writes a
tracked file.

The setting labels, help lines, chart names, axes and legends follow the developer's pattern
(a capital first, no articles, acronyms and the standard mathematical terms); every one is
listed in `docs/project/lab-words.md`.
