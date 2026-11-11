# Lab words

Every word the Lab shows, as it shows it now. They follow your pattern from BESS and RCWS
(2026-11-11: "use acronyms and standard math scientific terminology"):

> All words begin with a capital letter.
> Acronyms whenever possible (RMSE, RMS).
> No articles: "Dataset run", not "the dataset run".
> Standard terminology: the literature's symbols for DeePC's settings (T_s, T_ini, N, Q, R,
> λ_g, λ_y, σ_y, g, J), σ for a standard deviation.

**To change a word,** edit it here and tell me; I apply it. A chart's or a number's code name
(`output-e-lateral`, `rmse_position`) is not a word the page shows as a title: it stays the
file a save writes and a link's address key. Hovers are sentences: they begin with a capital and
keep their articles where a sentence needs them.

**Open, for your word:** a channel's name is the plant's own code name, used as it is in a chart's
title, ordinate and legend: the bicycle's e_lateral, e_heading, e_velocity, steering,
acceleration; the coupled masses' e_x1, e_v1, e_x2, e_v2, F1, F2. Giving them other words
(e_lat, δ, a, …) means each plant declares its own, a change in `tools/plants` I have not made.

## 1. Page

- header: DeePC Lab · the pieces' names: Closed loop · Dataset
- job state: Idle · Running… · Done · Ended with exit N
- header buttons: Stop · Refresh · New run; the folder beside them: results/closed_loop · N records
- headings: Runs · Compare · Saved stills · Figures; beside Compare's settings choice: Settings
- run list: Filter runs (empty box) · All · None; the count: N on disk · N shown · N listed
- a run's row: Use (hover: Load this run's settings into the form) · ✕ (hover: Delete this run's record, after asking)
- empty lists: No records yet: <the command that writes one> · No run holds every word typed ·
  No run shown: tick one above · Nothing saved yet: Save on a card writes its still here ·
  Tick a run to see what it can draw · Tick a run to see its charts ·
  Open a chart above, or click a number in the compare table
- compare: None · Differing · All; the corner: Column by column; the count: N numbers; tick
  another run to compare · N numbers, none of them differs · N settings · N numbers differ;
  the headings of a run's resolved values: Constants (each `group · key` as the record holds it,
  e.g. deepc_config · lambda_g)
- chart list: Show all · Hide all · N of M open; under a toggle: N of M runs · No shown run draws it;
  a group's index entry: ▸ Outputs · ▾ Outputs
- saved stills: N folders; a folder's line: <date> · N stills and N tables
- delete dialog: Delete <path>? · Delete · Cancel; after: Deleted … · Not deleted: …
- a card: Save · ✕ (hover: Close this card); its foot after Save: Writing… · Wrote … ·
  The draw wrote nothing; the terminal says why
- a number card's title: <number> vs <setting>, e.g. Position RMSE vs λ_g; its foot under one
  run: One run shown, or nothing differs between the shown runs: tick runs that differ in a setting
- terminal drawer: ▸ Terminal · ▾ Terminal; before any run: Nothing has been run in this session.
- messages: A run is in progress — stop it first · Stopping… · N settings loaded from <run> ·
  Closed loop: wrote N records · … stopped after it wrote N records · … the run was refused

## 2. Card controls

- bottom axis: X, offering the chart's own axis (e.g. Time [s]) or Sample index [-]
- logarithmic axes: Log x · Log y
- colour by: Colour by: Run · Quantity; on a number card, one toggle per setting that tells the runs apart, by its label
- runs on a chart: Overlay · Split · Mean · RMS; on a number card: All · Mean · RMS
- drawing: Dots · Lines; view: Chart · Table
- a table in a card's place: Six significant digits here; Save writes every digit to a CSV beside the still
- greyed, with the reason on the hover: One run shown · The pooled runs are one line per quantity ·
  Split: the colour tells the quantities apart · Nothing differs between the shown runs ·
  Nothing else tells the runs apart · No colour holds two runs at one value · and the figure
  layer's own reasons, begun with a capital: The chart declares no statistic · Its bottom axis
  holds N value(s) at or below zero · The runs draw different traces · …
- a card that cannot draw says why in its foot, in the words of what refused it, e.g. First
  declare no panel 'cost'; drawn without them, the comparison would leave them out unsaid
- legends: a run alone is its name; a trace every run holds identically: a, b — Reference
  (identical); a pooled line: Mean of 2 runs · e_lateral

## 3. Form

- drawer: New run · From defaults · From <run> · Clear · Close · Name · Sweep (none, then each
  setting's label) · v1, v2, … · Run
- a setting the code resolves (None in the settings): its empty box reads Code default, a pick's
  first option Code default; a setting with a declared default: its value, a pick's first option
  <value> (default)
- the dataset run's pick: default (default) · each saved dataset run by its name, its path on the line
- a group's note: N at defaults · N of M changed; a folded group's heading: ▸ Noise-adaptive · ▾ Noise-adaptive (and Online Hankel)
- under the bar: <names> exists — Run overwrites it; the line Run would type: lab: uv run deepc run …
- the note: N set — <setting> taken back: the run would ignore it · the sweep repeats a value —
  the terminal refuses the run · <setting> is set, and only some sweep values open it
- a greyed setting's hover: <setting> applies only when <setting> is <value>; it is <value> ·
  <setting> is the sweep axis; its values come from the sweep row
- a bound in a hover: [0 … ∞)

## 4. The pieces' own words

### Closed loop (`closed_loop`)

Settings, by group -- `code name`: label [unit] -- help line:

- **Data**
  - `dataset_run`: Dataset run -- The saved dataset run the controller's Hankel matrices are built from; run again first and checked against the file; its plant is the plant
- **Scenario**
  - `scenario`: Scenario -- The reference the plant is asked to follow; one the plant offers
  - `sim_duration`: Duration [s] -- Closed-loop duration in seconds; none: the code's own default
- **Controller**
  - `Tini`: T_ini [samples] -- Past window length in samples; none: the code's own default
  - `N`: N [samples] -- Prediction horizon in samples; none: the code's own default
  - `Q_diag`: Q [-] -- Output weights, one per output; none: the code's own default
  - `R_diag`: R [-] -- Input weights, one per input; none: the code's own default
  - `lambda_g`: λ_g [-] -- Weight of the regulariser on g; none: the code's own default
  - `lambda_y`: λ_y [-] -- Weight of the slack σ_y on the past outputs; none: the code's own default
  - `lambda_out`: λ_out [-] -- Weight of the slack on the output bounds; 0: no output bound; none: the code's own default
  - `reg_norm_g`: Norm on g [-] -- Regulariser norm on g (L2: the squared 2-norm, D14); none: the code's own default
  - `reg_norm_sigma_y`: Norm on σ_y [-] -- Norm on the past-output slack (L2: the squared 2-norm); none: the code's own default
- **Features**
  - `noise_adaptive`: Noise-adaptive λ -- Rescale the regularisation each step from the prediction residuals (D16)
  - `online_hankel`: Online Hankel -- Slide the Hankel window with closed-loop data (D5, D17)
  - `no_constraints`: No constraints -- Drop every input, rate and output bound from the controller; the plant still clips
- **Noise-adaptive** (folded)
  - `noise_estimation_window`: W [-] -- Residuals kept by the noise estimator; none: the code's own default
  - `baseline_noise_std`: σ_0 [-] -- Noise level the scaling divides by; none: the code's own default
  - `max_lambda_scaling`: k_max [-] -- Largest scaling of λ_g and λ_y; none: the code's own default
  - `constraint_tightening_factor`: c [-] -- Output-bound tightening per estimated std (D11); none: the code's own default
- **Online Hankel** (folded)
  - `hankel_window_size`: Hankel columns [-] -- Columns the online window holds; 0: the offline count; none: the code's own default
  - `hankel_warmup_steps`: Warm-up steps [-] -- Steps before the window starts sliding; none: the code's own default

### Dataset (`dataset`)

Settings, by group -- `code name`: label [unit] -- help line:

- **Plant**
  - `plant`: Plant -- The simulated plant the data is collected from
  - `Ts`: T_s [s] -- The plant's step and the data's sampling time, in seconds; the closed loop on this run steps at it too; none: the plant's own
- **Data**
  - `T_data`: T [samples] -- Samples collected, split across the plant's episodes; none: the plant's own
  - `noise_std_output`: σ_n [output unit] -- Standard deviation of the Gaussian noise added to each recorded output; none: the plant's own; in each output's own unit
- **Run**
  - `seed`: Seed [-] -- The seed of the episodes' initial states, the excitation and the noise

Charts of Closed loop (bicycle run), by group -- `code name`: title · bottom axis · ordinate · legend -- hover:

- **Outputs**
  - `output-e-lateral`: e_lateral · Time [s] · e_lateral [m] · e_lateral -- e_lateral at each step, warm-up included
  - `output-e-heading`: e_heading · Time [s] · e_heading [rad] · e_heading -- e_heading at each step, warm-up included
  - `output-e-velocity`: e_velocity · Time [s] · e_velocity [m/s] · e_velocity -- e_velocity at each step, warm-up included
- **Inputs**
  - `input-steering`: steering · Time [s] · steering [rad] · steering -- The controller's steering at each step, warm-up included, before the plant's own clipping
  - `input-acceleration`: acceleration · Time [s] · acceleration [m/s²] · acceleration -- The controller's acceleration at each step, warm-up included, before the plant's own clipping
- **Solver**
  - `solve-time`: Solve time · Solve [-] · Solve time [s] · Solve time -- Wall-clock time of the solve at each solve
  - `cost`: J · Solve [-] · J [-] · J -- Objective value J at each solve
  - `sigma-y-norm`: ‖σ_y‖₂ · Solve [-] · ‖σ_y‖₂ [-] · ‖σ_y‖₂ -- 2-norm of the past-output slack σ_y at each solve
  - `lambda-g`: λ_g · Solve [-] · λ_g [-] · λ_g -- λ_g the solve used at each solve
- **Position**
  - `position-x`: Position x · Step [-] · x [m] · Plant, Reference -- The plant's x after each step beside the waypoint of the same index, as the position RMSE compares them
  - `position-y`: Position y · Step [-] · y [m] · Plant, Reference -- The plant's y after each step beside the waypoint of the same index, as the position RMSE compares them

Numbers of Closed loop (bicycle run), by group -- `code name`: label [unit] -- hover:

- **Outputs**
  - `rmse_e_lateral`: e_lateral RMSE [m] -- Root mean square of e_lateral over every step, warm-up included
  - `rmse_e_heading`: e_heading RMSE [rad] -- Root mean square of e_heading over every step, warm-up included
  - `rmse_e_velocity`: e_velocity RMSE [m/s] -- Root mean square of e_velocity over every step, warm-up included
  - `rmse_x`: X RMSE [m] -- The plant's own metric (compute_custom_metrics)
  - `rmse_y`: Y RMSE [m] -- The plant's own metric (compute_custom_metrics)
  - `rmse_position`: Position RMSE [m] -- The plant's own metric (compute_custom_metrics)
- **Solver**
  - `solves_optimal_inaccurate`: Solves: optimal inaccurate -- Solves whose status is exactly 'optimal_inaccurate'
  - `solves_optimal`: Solves: optimal -- Solves whose status is exactly 'optimal'
  - `solves_by_OSQP`: Solves by OSQP -- Solves OSQP answered (`solves_by_SCS`: Solves by SCS -- Solves SCS answered, after OSQP raised a SolverError)
  - `zero_inputs_applied`: Zero inputs applied -- Solves that left no solution, after which the controller's zero input was applied
  - `solve_time_mean_s`: Mean solve time [s] -- Mean wall-clock time of a solve
  - `solve_time_max_s`: Max solve time [s] -- Longest wall-clock time of a solve
  - `sigma_y_norm_mean`: Mean ‖σ_y‖₂ -- Mean 2-norm of the past-output slack σ_y
  - `sigma_y_norm_max`: Max ‖σ_y‖₂ -- Largest 2-norm of the past-output slack σ_y

Charts of Dataset (bicycle run), by group -- `code name`: title · bottom axis · ordinate · legend -- hover:

- **Inputs**
  - `input-steering`: steering · Sample [-] · steering [rad] · steering -- The recorded steering at each sample, episodes stacked end to end
  - `input-acceleration`: acceleration · Sample [-] · acceleration [m/s²] · acceleration -- The recorded acceleration at each sample, episodes stacked end to end
- **Outputs**
  - `output-e-lateral`: e_lateral · Sample [-] · e_lateral [m] · e_lateral -- The recorded e_lateral at each sample, episodes stacked end to end
  - `output-e-heading`: e_heading · Sample [-] · e_heading [rad] · e_heading -- The recorded e_heading at each sample, episodes stacked end to end
  - `output-e-velocity`: e_velocity · Sample [-] · e_velocity [m/s] · e_velocity -- The recorded e_velocity at each sample, episodes stacked end to end

Numbers of Dataset (bicycle run), by group -- `code name`: label [unit] -- hover:

- **Inputs**
  - `u_rms_steering`: steering RMS [rad] -- Root mean square of the recorded steering over all samples
  - `u_rms_acceleration`: acceleration RMS [m/s²] -- Root mean square of the recorded acceleration over all samples
- **Outputs**
  - `y_rms_e_lateral`: e_lateral RMS [m] -- Root mean square of the recorded e_lateral over all samples
  - `y_rms_e_heading`: e_heading RMS [rad] -- Root mean square of the recorded e_heading over all samples
  - `y_rms_e_velocity`: e_velocity RMS [m/s] -- Root mean square of the recorded e_velocity over all samples
