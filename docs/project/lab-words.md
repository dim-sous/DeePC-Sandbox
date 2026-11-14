# Lab words

Every word the Lab shows, as it shows it now. They follow your pattern from BESS and RCWS
(2026-11-11: "use acronyms and standard math scientific terminology"):

> All words begin with a capital letter.
> Acronyms whenever possible (RMSE, RMS).
> No articles: "Dataset run", not "the dataset run".
> Standard terminology: the literature's symbols for DeePC's settings (T_s, T_ini, N, Q, R,
> λ_g, λ_y, σ_y, g, J), σ for a standard deviation.

**To change a word,** edit it here and tell me; I apply it. A piece's words (its settings, charts
and numbers) are written in `devkit/harness/`, and so are the page's title and the command
whose line it prints beside each run (`TITLE`, `COMMAND` in `devkit/harness/__init__.py`). The page's other words (its buttons,
headings and refusals, the charts' refusals among them) are written in Benchlab, which DeePC
installs, until Benchlab turns them into cells this repository fills (its `docs/design.md`,
decision 4): a change to one of those is made in Benchlab and reaches this page with Benchlab's
next pinned tag. A chart's or a number's code name
(`output-e-x1`, `rmse_e_x1`) is not a word the page shows as a title: it stays the
file `deepc draw` writes and a link's address key. No word is on a hover but a greyed control's
reason and a chart's own -- a point's run and value, and the chart toolbar's button names (Zoom,
Pan, Zoom in, Zoom out, Reset axes), plotly's -- (Benchlab's `docs/design.md` decision 5, since
Benchlab 0.3.0); a reason is a sentence: it begins with a capital and keeps its articles where a
sentence needs them.

**Your word, 2026-11-11:** inputs (manipulated variables) are u1…, outputs (measurements) y1…,
true states x1…, time t — "the names everywhere (axis labels, chart buttons, EVERYWHERE)" — and
no units anywhere. The System piece follows it; the Dataset and Closed loop pieces keep the plant
classes' names (e_x1, F1, …) and units until the rewrite replaces them (`docs/project/design.md`).

## 1. Page

- header: DeePC Lab · the pieces' names: Closed loop · Dataset · System · Open loop
- header buttons: Refresh · New run; the folder beside them: results/closed_loop · N records
- headings: Runs · Compare · Figures; beside Compare's settings choice: Settings
- run list: Filter runs (empty box) · All · None; the count: N on disk · N shown · N listed
- a run's row: Use · ✕
- empty lists: No records yet: <the command that writes one> · No run holds every word typed ·
  No run shown: tick one above ·
  Tick a run to see what it can draw · Tick a run to see its charts ·
  Open a chart above, or click a number in the compare table
- compare: None · Differing · All; the corner: Column by column; the count: N numbers; tick
  another run to compare · N numbers, none of them differs · N settings · N numbers differ;
  the headings of a run's resolved values: Constants (each `group · key` as the record holds it,
  e.g. deepc_config · lambda_g)
- chart list: Show all · Hide all · N of M open; under a toggle: N of M runs · No shown run draws it;
  a group's index entry: ▸ Outputs · ▾ Outputs
- delete dialog: Delete <path>? · Delete · Cancel; after: Deleted … · Not deleted: …
- a card: ✕
- a number card's title: <number> vs <setting>, e.g. Position RMSE vs λ_g; its foot under one
  run: One run shown, or nothing differs between the shown runs: tick runs that differ in a setting
- runs in progress, along the bottom: ▸ Runs in progress · ▾ Runs in progress; beside it, the
  count in this order, each only when there is one: N running · N queued · N done · N failed ·
  N stopped (Nothing running before any run), and up to N side by side; a run's row: <name> ·
  <piece>, its state Running m:ss · Queued · Done m:ss · Failed m:ss with the words of what
  failed it · Stopped (m:ss once it had started), and Stop, or ✕ once it ended
- a failed run's words, when they are not its own: ended with exit N · ended by signal SIGKILL
  (or another signal's name) · could not start: <words> · its answer could not be read: <words>
  · its end could not be read: <words>
- messages: Stopping… · N settings loaded from <run> · Deleted <path> · Not deleted: <words> ·
  reading the runs failed: <words>

## 2. Card controls

A chart card has one control, a number card two (the developer's cut, 2026-11-11):

- runs on a chart: Runs: Overlay · Split · Mean · RMS
- on a number card: X, offering the settings that differ between the shown runs, by their label;
  Runs: All · Mean · RMS
- greyed, with the reason on the hover: One run shown · Nothing differs between the shown runs ·
  No colour holds two runs at one value · and the figure layer's own reasons, begun with a
  capital: The chart declares no statistic · The runs draw different traces · …
- a card that cannot draw says why in its foot, in the words of what refused it, e.g. First
  declare no panel 'cost'; drawn without them, the comparison would leave them out unsaid
- legends: a run alone is its name; a trace every run holds identically: a, b — e_x1
  (identical); a pooled line: Mean of 2 runs · e_x1

## 3. Form

- drawer: New run · From defaults · From <run> · Clear · Close · Name · Sweep (none, then each
  setting's label) · v1, v2, … · Run
- a setting the code resolves (None in the settings): its empty box reads Code default, a pick's
  first option Code default; a setting with a declared default: its value, a pick's first option
  <value> (default)
- the dataset run's pick: default (default) · each saved dataset run by its name
- a group's note: N at defaults · N of M changed; a folded group's heading: ▸ Noise-adaptive · ▾ Noise-adaptive (and Online Hankel)
- under the bar: <names> exists — Run overwrites it; why Run would be refused: refused: <words>,
  e.g. refused: <name> is queued or running already; give this run another name
- the note: N set — <setting> taken back: the run would ignore it · the sweep repeats a value —
  Run refuses it · <setting> is set, and only some sweep values open it
- under a greyed setting: <setting> applies only when <setting> is <value>; it is <value> ·
  <setting> is the sweep axis; its values come from the sweep row
- under a setting, in red, when Run refuses its value (nothing is queued): <setting> must be at
  least <low> | above <low> | at most <high> | below <high>, got <text> · <setting> takes a
  number | a whole number | a list of numbers | a list of texts | true or false | one of <its
  choices>, got <text>; under the sweep row, the same for the first sweep value refused; a
  refused setting's folded group opens

## 4. The pieces' own words

### Closed loop (`closed_loop`)

Settings, by group -- `code name`: label [unit]:

- **Data**
  - `dataset_run`: Dataset run
- **Scenario**
  - `scenario`: Scenario
  - `sim_duration`: Duration [s]
- **Controller**
  - `Tini`: T_ini [samples]
  - `N`: N [samples]
  - `Q_diag`: Q [-]
  - `R_diag`: R [-]
  - `lambda_g`: λ_g [-]
  - `lambda_y`: λ_y [-]
  - `lambda_out`: λ_out [-]
  - `reg_norm_g`: Norm on g [-]
  - `reg_norm_sigma_y`: Norm on σ_y [-]
- **Features**
  - `noise_adaptive`: Noise-adaptive λ
  - `online_hankel`: Online Hankel
  - `no_constraints`: No constraints
- **Noise-adaptive** (folded)
  - `noise_estimation_window`: W [-]
  - `baseline_noise_std`: σ_0 [-]
  - `max_lambda_scaling`: k_max [-]
  - `constraint_tightening_factor`: c [-]
- **Online Hankel** (folded)
  - `hankel_window_size`: Hankel columns [-]
  - `hankel_warmup_steps`: Warm-up steps [-]

### Dataset (`dataset`)

Settings, by group -- `code name`: label [unit]:

- **Plant**
  - `plant`: Plant
  - `Ts`: T_s [s]
- **Data**
  - `T_data`: T [samples]
  - `noise_std_output`: σ_n [output unit]
- **Run**
  - `seed`: Seed [-]

Charts of Closed loop (coupled masses run), by group -- `code name`: title · bottom axis · ordinate · legend:

- **Outputs**
  - `output-e-x1`: e_x1 · Time [s] · e_x1 [m] · e_x1
  - `output-e-v1`: e_v1 · Time [s] · e_v1 [m/s] · e_v1
  - `output-e-x2`: e_x2 · Time [s] · e_x2 [m] · e_x2
  - `output-e-v2`: e_v2 · Time [s] · e_v2 [m/s] · e_v2
- **Inputs**
  - `input-F1`: F1 · Time [s] · F1 [N] · F1, Lower bound of F1 [N], Upper bound of F1 [N]
  - `input-F2`: F2 · Time [s] · F2 [N] · F2, Lower bound of F2 [N], Upper bound of F2 [N]
- **Solver**
  - `solve-time`: Solve time · Solve [-] · Solve time [s] · Solve time
  - `cost`: J · Solve [-] · J [-] · J
  - `sigma-y-norm`: ‖σ_y‖₂ · Solve [-] · ‖σ_y‖₂ [-] · ‖σ_y‖₂
  - `lambda-g`: λ_g · Solve [-] · λ_g [-] · λ_g

Numbers of Closed loop (coupled masses run), by group -- `code name`: label [unit]:

- **Outputs**
  - `rmse_e_x1`: e_x1 RMSE [m]
  - `rmse_e_v1`: e_v1 RMSE [m/s]
  - `rmse_e_x2`: e_x2 RMSE [m]
  - `rmse_e_v2`: e_v2 RMSE [m/s]
  - `rmse_position_combined`: Position combined RMSE [m]
- **Solver**
  - `solves_optimal_inaccurate`: Solves: optimal inaccurate
  - `solves_optimal`: Solves: optimal
  - `solves_by_OSQP`: Solves by OSQP (`solves_by_SCS`: Solves by SCS)
  - `zero_inputs_applied`: Zero inputs applied
  - `solve_time_mean_s`: Mean solve time [s]
  - `solve_time_max_s`: Max solve time [s]
  - `sigma_y_norm_mean`: Mean ‖σ_y‖₂
  - `sigma_y_norm_max`: Max ‖σ_y‖₂

Charts of Dataset (coupled masses run), by group -- `code name`: title · bottom axis · ordinate · legend:

- **Inputs**
  - `input-F1`: F1 · Sample [-] · F1 [N] · F1, Lower bound of F1 [N], Upper bound of F1 [N]
  - `input-F2`: F2 · Sample [-] · F2 [N] · F2, Lower bound of F2 [N], Upper bound of F2 [N]
- **Outputs**
  - `output-e-x1`: e_x1 · Sample [-] · e_x1 [m] · e_x1
  - `output-e-v1`: e_v1 · Sample [-] · e_v1 [m/s] · e_v1
  - `output-e-x2`: e_x2 · Sample [-] · e_x2 [m] · e_x2
  - `output-e-v2`: e_v2 · Sample [-] · e_v2 [m/s] · e_v2

Numbers of Dataset (coupled masses run), by group -- `code name`: label [unit]:

- **Inputs**
  - `u_rms_F1`: F1 RMS [N]
  - `u_rms_F2`: F2 RMS [N]
- **Outputs**
  - `y_rms_e_x1`: e_x1 RMS [m]
  - `y_rms_e_v1`: e_v1 RMS [m/s]
  - `y_rms_e_x2`: e_x2 RMS [m]
  - `y_rms_e_v2`: e_v2 RMS [m/s]

### System (`system`)

Settings, by group -- `code name`: label (· its rows · add button) (no units):

- **States**
  - `f`: x' · rows x1' =, x2' =, … · + State
- **Outputs**
  - `h`: y · rows y1 =, y2 =, … · + Output
- **Parameters**
  - `theta`: θ · rows θ1 =, θ2 =, … · + Parameter
- **Analysis**
  - `x_star`: x*
  - `u_star`: u*
  - `input`: Input
  - `output`: Output

Charts of System (default run, the two masses at the origin, u1 → y1), by group -- `code name`: title · bottom axis · ordinate · legend:

- **Frequency**
  - `bode-magnitude`: Bode magnitude · ω · |G| · |G| of u1 → y1 (log ω, log |G|)
  - `bode-phase`: Bode phase · ω · ∠G · ∠G of u1 → y1 (log ω)
- **Time**
  - `step`: Step response · t · δy1 · δy1
  - `impulse`: Impulse response · t · δy1 · δy1
  - `ramp`: Ramp response · t · δy1 · δy1

Numbers of System, by group -- `code name`: label:

- **Sizes**
  - `n`: n
  - `m`: m
  - `p`: p
  - `q`: q
  - `r`: r
- **Poles**
  - `unstable_poles`: Unstable poles
  - `max_pole_re`: Max pole Re
  - `min_damping`: Min damping ratio
  - `residual`: Equilibrium residual
- **Pair**
  - `dc_gain`: DC gain
  - `gain_margin`: Gain margin
  - `phase_margin`: Phase margin


### Open loop (`open_loop`)

Settings, by group -- `code name`: label (no units):

- **System**
  - `system_run`: System run
  - `x0`: x_0
- **Inputs**
  - `u_min`: u_min
  - `u_max`: u_max
  - `u_exc_min`: u_exc_min
  - `u_exc_max`: u_exc_max
- **Simulation**
  - `Ts`: T_s
  - `T`: T
  - `sigma`: σ
  - `sigma_d`: σ_d
  - `seed`: Seed
- **Integrator** (folded)
  - `rtol`: rtol
  - `atol`: atol

Charts of Open loop (default run, the two masses), by group -- `code name`: title · bottom axis · ordinate · legend:

- **Inputs**
  - `u1`: u1 · t · u1 · u1, Lower bound of u1, Upper bound of u1
  - `u2`: u2 · t · u2 · u2, Lower bound of u2, Upper bound of u2
- **Outputs**
  - `y1`: y1 · t · y1 · y1
  - `y2`: y2 · t · y2 · y2
  - `y3`: y3 · t · y3 · y3
  - `y4`: y4 · t · y4 · y4
- **States**
  - `x1`: x1 · t · x1 · x1
  - `x2`: x2 · t · x2 · x2
  - `x3`: x3 · t · x3 · x3
  - `x4`: x4 · t · x4 · x4
- **Disturbances**
  - `d1`: d1 · t · d1 · d1
  - `d2`: d2 · t · d2 · d2

Numbers of Open loop, by group -- `code name`: label:

- **Inputs**
  - `pe_order`: PE order of u
  - `clipped`: Clipped inputs
- **Outputs**
  - `y1_snr`: y1 SNR
  - `y2_snr`: y2 SNR
  - `y3_snr`: y3 SNR
  - `y4_snr`: y4 SNR
- **Cost**
  - `step_time_mean`: Mean step time
  - `step_time_std`: Step time std
  - `step_time_max`: Max step time
