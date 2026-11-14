# Lab words

Every word the Lab shows, as it shows it now. They follow your pattern from BESS and RCWS
(2026-11-11: "use acronyms and standard math scientific terminology"):

> All words begin with a capital letter.
> Acronyms whenever possible (RMSE, RMS).
> No articles: "Dataset run", not "the dataset run".
> Standard terminology: the literature's symbols for DeePC's settings (T_s, T_ini, N, Q, R,
> λ_g, λ_y, σ_y, g, J), σ for a standard deviation.

**To change a word,** edit it here and tell me; I apply it. A piece's words (its settings, charts
and numbers) are written in `devkit/harness/`, and so are the page's title and the command it
launches (`TITLE`, `COMMAND` in `devkit/harness/__init__.py`). The page's other words (its buttons,
headings and refusals, the charts' refusals among them) are written in Benchlab, which DeePC
installs, until Benchlab turns them into cells this repository fills (its `docs/design.md`,
decision 4): a change to one of those is made in Benchlab and reaches this page with Benchlab's
next pinned tag. A chart's or a number's code name
(`output-e-x1`, `rmse_e_x1`) is not a word the page shows as a title: it stays the
file `deepc draw` writes and a link's address key. Hovers are sentences: they begin with a capital and
keep their articles where a sentence needs them.

**Your word, 2026-11-11:** inputs (manipulated variables) are u1…, outputs (measurements) y1…,
true states x1…, time t — "the names everywhere (axis labels, chart buttons, EVERYWHERE)" — and
no units anywhere. The System piece follows it; the Dataset and Closed loop pieces keep the plant
classes' names (e_x1, F1, …) and units until the rewrite replaces them (`docs/project/design.md`).

## 1. Page

- header: DeePC Lab · the pieces' names: Closed loop · Dataset · System · Open loop
- job state: Idle · Running… · Done · Ended with exit N
- header buttons: Stop · Refresh · New run; the folder beside them: results/closed_loop · N records
- headings: Runs · Compare · Figures; beside Compare's settings choice: Settings
- run list: Filter runs (empty box) · All · None; the count: N on disk · N shown · N listed
- a run's row: Use (hover: Load this run's settings into the form) · ✕ (hover: Delete this run's record, after asking)
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
- a card: ✕ (hover: Close this card)
- a number card's title: <number> vs <setting>, e.g. Position RMSE vs λ_g; its foot under one
  run: One run shown, or nothing differs between the shown runs: tick runs that differ in a setting
- terminal drawer: ▸ Terminal · ▾ Terminal; before any run: Nothing has been run in this session.
- messages: A run is in progress — stop it first · Stopping… · N settings loaded from <run> ·
  Closed loop: wrote N records · … stopped after it wrote N records · … the run was refused

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

Charts of Closed loop (coupled masses run), by group -- `code name`: title · bottom axis · ordinate · legend -- hover:

- **Outputs**
  - `output-e-x1`: e_x1 · Time [s] · e_x1 [m] · e_x1 -- e_x1 at each step, warm-up included
  - `output-e-v1`: e_v1 · Time [s] · e_v1 [m/s] · e_v1 -- e_v1 at each step, warm-up included
  - `output-e-x2`: e_x2 · Time [s] · e_x2 [m] · e_x2 -- e_x2 at each step, warm-up included
  - `output-e-v2`: e_v2 · Time [s] · e_v2 [m/s] · e_v2 -- e_v2 at each step, warm-up included
- **Inputs**
  - `input-F1`: F1 · Time [s] · F1 [N] · F1, Lower bound of F1 [N], Upper bound of F1 [N] -- The controller's F1 at each step, warm-up included, before the plant's own clipping; the plant's bounds on it
  - `input-F2`: F2 · Time [s] · F2 [N] · F2, Lower bound of F2 [N], Upper bound of F2 [N] -- The controller's F2 at each step, warm-up included, before the plant's own clipping; the plant's bounds on it
- **Solver**
  - `solve-time`: Solve time · Solve [-] · Solve time [s] · Solve time -- Wall-clock time of the solve at each solve
  - `cost`: J · Solve [-] · J [-] · J -- Objective value J at each solve
  - `sigma-y-norm`: ‖σ_y‖₂ · Solve [-] · ‖σ_y‖₂ [-] · ‖σ_y‖₂ -- 2-norm of the past-output slack σ_y at each solve
  - `lambda-g`: λ_g · Solve [-] · λ_g [-] · λ_g -- λ_g the solve used at each solve

Numbers of Closed loop (coupled masses run), by group -- `code name`: label [unit] -- hover:

- **Outputs**
  - `rmse_e_x1`: e_x1 RMSE [m] -- Root mean square of e_x1 over every step, warm-up included
  - `rmse_e_v1`: e_v1 RMSE [m/s] -- Root mean square of e_v1 over every step, warm-up included
  - `rmse_e_x2`: e_x2 RMSE [m] -- Root mean square of e_x2 over every step, warm-up included
  - `rmse_e_v2`: e_v2 RMSE [m/s] -- Root mean square of e_v2 over every step, warm-up included
  - `rmse_position_combined`: Position combined RMSE [m] -- The plant's own metric (compute_custom_metrics)
- **Solver**
  - `solves_optimal_inaccurate`: Solves: optimal inaccurate -- Solves whose status is exactly 'optimal_inaccurate'
  - `solves_optimal`: Solves: optimal -- Solves whose status is exactly 'optimal'
  - `solves_by_OSQP`: Solves by OSQP -- Solves OSQP answered (`solves_by_SCS`: Solves by SCS -- Solves SCS answered, after OSQP raised a SolverError)
  - `zero_inputs_applied`: Zero inputs applied -- Solves that left no solution, after which the controller's zero input was applied
  - `solve_time_mean_s`: Mean solve time [s] -- Mean wall-clock time of a solve
  - `solve_time_max_s`: Max solve time [s] -- Longest wall-clock time of a solve
  - `sigma_y_norm_mean`: Mean ‖σ_y‖₂ -- Mean 2-norm of the past-output slack σ_y
  - `sigma_y_norm_max`: Max ‖σ_y‖₂ -- Largest 2-norm of the past-output slack σ_y

Charts of Dataset (coupled masses run), by group -- `code name`: title · bottom axis · ordinate · legend -- hover:

- **Inputs**
  - `input-F1`: F1 · Sample [-] · F1 [N] · F1, Lower bound of F1 [N], Upper bound of F1 [N] -- The recorded F1 at each sample, episodes stacked end to end; the plant's bounds on it
  - `input-F2`: F2 · Sample [-] · F2 [N] · F2, Lower bound of F2 [N], Upper bound of F2 [N] -- The recorded F2 at each sample, episodes stacked end to end; the plant's bounds on it
- **Outputs**
  - `output-e-x1`: e_x1 · Sample [-] · e_x1 [m] · e_x1 -- The recorded e_x1 at each sample, episodes stacked end to end
  - `output-e-v1`: e_v1 · Sample [-] · e_v1 [m/s] · e_v1 -- The recorded e_v1 at each sample, episodes stacked end to end
  - `output-e-x2`: e_x2 · Sample [-] · e_x2 [m] · e_x2 -- The recorded e_x2 at each sample, episodes stacked end to end
  - `output-e-v2`: e_v2 · Sample [-] · e_v2 [m/s] · e_v2 -- The recorded e_v2 at each sample, episodes stacked end to end

Numbers of Dataset (coupled masses run), by group -- `code name`: label [unit] -- hover:

- **Inputs**
  - `u_rms_F1`: F1 RMS [N] -- Root mean square of the recorded F1 over all samples
  - `u_rms_F2`: F2 RMS [N] -- Root mean square of the recorded F2 over all samples
- **Outputs**
  - `y_rms_e_x1`: e_x1 RMS [m] -- Root mean square of the recorded e_x1 over all samples
  - `y_rms_e_v1`: e_v1 RMS [m/s] -- Root mean square of the recorded e_v1 over all samples
  - `y_rms_e_x2`: e_x2 RMS [m] -- Root mean square of the recorded e_x2 over all samples
  - `y_rms_e_v2`: e_v2 RMS [m/s] -- Root mean square of the recorded e_v2 over all samples

### System (`system`)

Settings, by group -- `code name`: label (· its rows · add button) -- help line (no units):

- **States**
  - `f`: x' · rows x1' =, x2' =, … · + State -- One row per state: x_i' as an expression in x1…, u1…, the disturbances d1…, theta1… and sin cos tan exp log sqrt tanh abs pi
- **Outputs**
  - `h`: y · rows y1 =, y2 =, … · + Output -- One row per output: y_j as an expression in x1…, theta1… and the same functions
- **Parameters**
  - `theta`: θ · rows θ1 =, θ2 =, … · + Parameter -- One row per parameter: the value of theta_k, which the rows name as theta1, theta2, …
- **Analysis**
  - `x_star`: x* -- The state to linearise at, one value per state; none: the origin
  - `u_star`: u* -- The input to linearise at, one value per input; none: zero
  - `input`: Input -- The input u_j of the analysed pair, by its index j
  - `output`: Output -- The output y_i of the analysed pair, by its index i

Charts of System (default run, the two masses at the origin, u1 → y1), by group -- `code name`: title · bottom axis · ordinate · legend -- hover:

- **Frequency**
  - `bode-magnitude`: Bode magnitude · ω · |G| · |G| of u1 → y1 (log ω, log |G|) -- The magnitude of the frequency response of u1 → y1 at (x*, u*)
  - `bode-phase`: Bode phase · ω · ∠G · ∠G of u1 → y1 (log ω) -- The phase of the frequency response of u1 → y1 at (x*, u*), in degrees
- **Time**
  - `step`: Step response · t · δy1 · δy1 -- y1's deviation from h(x*) after a unit step in u1, linearised at (x*, u*)
  - `impulse`: Impulse response · t · δy1 · δy1 -- y1's deviation from h(x*) after a unit impulse in u1, linearised at (x*, u*)
  - `ramp`: Ramp response · t · δy1 · δy1 -- y1's deviation from h(x*) under u1 - u1* = t, linearised at (x*, u*)

Numbers of System, by group -- `code name`: label -- hover:

- **Sizes**
  - `n`: n -- The number of states, x1…xn: one per state row
  - `m`: m -- The number of inputs, u1…um
  - `p`: p -- The number of outputs, y1…yp: one per output row
  - `q`: q -- The number of parameters, theta1…thetaq
  - `r`: r -- The number of disturbance inputs, d1…dr: where the unmodelled disturbance enters
- **Poles**
  - `unstable_poles`: Unstable poles -- Eigenvalues of A with a positive real part, at (x*, u*)
  - `max_pole_re`: Max pole Re -- The largest real part of A's eigenvalues
  - `min_damping`: Min damping ratio -- The smallest damping ratio of A's eigenvalues (python-control's damp); a pole at the origin has none
  - `residual`: Equilibrium residual -- |f(x*, u*, 0)|: 0 when (x*, u*) is an equilibrium
- **Pair**
  - `dc_gain`: DC gain -- The pair's steady-state gain G(0)
  - `gain_margin`: Gain margin -- The pair's gain margin, as python-control's stability_margins computes it
  - `phase_margin`: Phase margin -- The pair's phase margin in degrees, as python-control's stability_margins computes it


### Open loop (`open_loop`)

Settings, by group -- `code name`: label -- help line (no units):

- **System**
  - `system_run`: System run -- The saved system run simulated; read again first and checked against the file
  - `x0`: x0 -- The initial state, one value per state
- **Inputs**
  - `u_min`: u_min -- The lower actuator bound of each input, which the plant clips u to; -inf for none
  - `u_max`: u_max -- The upper actuator bound of each input, which the plant clips u to; inf for none
  - `u_exc_min`: u_exc_min -- The lower end of each input's excitation, drawn i.i.d. uniform at every sample
  - `u_exc_max`: u_exc_max -- The upper end of each input's excitation, drawn i.i.d. uniform at every sample
- **Simulation**
  - `Ts`: T_s -- The sampling time: the input is held over it and the output measured at its ends
  - `T`: T -- The number of samples simulated, from x0
  - `sigma`: σ -- The standard deviation of the Gaussian noise added to every output at every sample
  - `sigma_d`: σ_d -- The standard deviation of each disturbance d1… the system's rows name, drawn Gaussian at every sample and held over T_s; [] for a system with none
  - `seed`: Seed -- The seed of the excitation, the measurement noise and the disturbances
- **Integrator** (folded)
  - `rtol`: rtol -- RK45's relative tolerance over each step
  - `atol`: atol -- RK45's absolute tolerance over each step

Charts of Open loop (default run, the two masses), by group -- `code name`: title · bottom axis · ordinate · legend -- hover:

- **Inputs**
  - `u1`: u1 · t · u1 · u1, Lower bound of u1, Upper bound of u1 -- The u1 the plant applied at each sample, held over T_s; its actuator bounds
  - `u2`: u2 · t · u2 · u2, Lower bound of u2, Upper bound of u2 -- The u2 the plant applied at each sample, held over T_s; its actuator bounds
- **Outputs**
  - `y1`: y1 · t · y1 · y1 -- y1 measured at each sample, noise included
  - `y2`: y2 · t · y2 · y2 -- y2 measured at each sample, noise included
  - `y3`: y3 · t · y3 · y3 -- y3 measured at each sample, noise included
  - `y4`: y4 · t · y4 · y4 -- y4 measured at each sample, noise included
- **States**
  - `x1`: x1 · t · x1 · x1 -- The true x1 at each sample, which the controller never reads
  - `x2`: x2 · t · x2 · x2 -- The true x2 at each sample, which the controller never reads
  - `x3`: x3 · t · x3 · x3 -- The true x3 at each sample, which the controller never reads
  - `x4`: x4 · t · x4 · x4 -- The true x4 at each sample, which the controller never reads
- **Disturbances**
  - `d1`: d1 · t · d1 · d1 -- The disturbance d1 held over each step, which the controller never reads
  - `d2`: d2 · t · d2 · d2 -- The disturbance d2 held over each step, which the controller never reads

Numbers of Open loop, by group -- `code name`: label -- hover:

- **Inputs**
  - `pe_order`: PE order of u -- The largest L at which the input's Hankel matrix H_L(u) has full row rank L·m: persistency of excitation of order L
  - `clipped`: Clipped inputs -- Input samples the actuator bounds changed, over all inputs
- **Outputs**
  - `y1_snr`: y1 SNR -- Variance of the noise-free y1, h(x), over the measurement noise's variance σ²
  - `y2_snr`: y2 SNR -- Variance of the noise-free y2, h(x), over the measurement noise's variance σ²
  - `y3_snr`: y3 SNR -- Variance of the noise-free y3, h(x), over the measurement noise's variance σ²
  - `y4_snr`: y4 SNR -- Variance of the noise-free y4, h(x), over the measurement noise's variance σ²
- **Cost**
  - `step_time_mean`: Mean step time -- Mean wall-clock seconds of one sampling step: the clip and the integration
  - `step_time_std`: Step time std -- Standard deviation of the wall-clock seconds of one sampling step
  - `step_time_max`: Max step time -- Longest wall-clock seconds of one sampling step
