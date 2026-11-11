# devkit/ — the test bench

Runs the controller (`src/`) on the simulated plants (`tools/`), records each run, draws the
records, and (later) shows them in the Lab. May import everything; nothing in `src/` or `tools/`
imports it (`tests/test_roots.py`). Who imports whom inside it is held by
`tests/test_devkit_layers.py`.

| package | holds |
|---|---|
| `sim/` | the simulation code the pieces drive, as `run.py` drives it: data collection (`collect_data`, `episode_lengths`), the scenarios (`get_reference`), the closed loop (`run_simulation`), the `DeePCConfig` built from a plant (`build_deepc_config`) |
| `record/` | the settings layer (`record.config.Settings` on `param`: every setting with a group, refused in words when wrong, `needs` for a setting that applies only while another holds a named value, `None` for "the code's own default", `record_of` for a setting naming another piece's saved run) and the record layer (`Record`, saved as one JSON and loaded back bit for bit; `Series` and `Number`, the charts and numbers a record declares; `Provenance`, what produced it) |
| `harness/` | the registry of pieces (`NAMES`, `get_component`: reading it imports no piece) and one package per piece, each exposing `Config`, `run(config) -> Record` and `Record`. `dataset`: training data from a plant, as `collect_data` makes it, episodes stacked end to end. `closed_loop`: DeePC on a saved dataset run (re-run first and checked against its file, `harness.dataset.dataset_of`), following one of its plant's scenarios; every solve's status, cost, slack and time kept, solves counted by status (D2), the plant's step beside the config's (D1) |
| `figure/` | stills over one or more records, drawn from one panel spec (`render.Panel`) that `chart` builds from what the records declare: a declared panel over one run (its traces in the series colours) or several on one plot (the colour carrying the run, the dash the trace, the legend naming "run — trace"), or a summary number against a setting, every run a point, coloured by the settings that tell the runs apart but a replicate (`chart.sweep_of`). Axis labels with units, a legend only when there is more than one label, no title; what it cannot draw it refuses in words. Every save goes through `style.save_figure`: the PNG and a JSON sidecar (commit, date, command, runs) |
| `terminal/` | `deepc run <piece>`: build the Config, apply each `--set`, run once per `--sweep` combination, save one record per run. `deepc draw <record>...`: the panels the records declare as PNGs beside them, or `--number N --vs SETTING` |

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
```

A setting left unset takes the code's own default, the plant's where it has one, and the record
saves what it resolved (`constants`). A record holds the settings the run used, the arrays its
charts draw, summary numbers each a reduction of one of them, and its provenance (commit,
dirty flag, date, duration, library versions). `results/` is git-ignored: a run never writes a
tracked file.

The setting labels, help lines and chart names are placeholders: the words a Lab user reads are
the developer's to choose, asked in the Lab branch (RCWS #96).
