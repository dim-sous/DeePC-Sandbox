# Lab words

Every word the Lab shows, as it shows it now. They follow your pattern from BESS and RCWS
(2026-11-11: "use acronyms and standard math scientific terminology"):

> All words begin with a capital letter.
> Acronyms whenever possible (RMSE, RMS).
> No articles: "System run", not "the system run".
> Standard terminology: the literature's symbols for DeePC's settings (T_s, T_ini, N, Q, R,
> λ_g, λ_y, σ_y, g, J), σ for a standard deviation.

**Where a word is written.** The page's own words -- its buttons, headings, counts, notes and
refusals -- are in `devkit/harness/words.toml`, one slot each that Benchlab names (Benchlab
0.4.0, its `docs/design.md` decision 4): to change one, change it there. A piece's words (its
settings, charts and numbers) are in `devkit/harness/<piece>/`, listed below (§2); the page's
title and the command whose line it prints beside each run are `TITLE` and `COMMAND` in
`devkit/harness/__init__.py`. The words of what refused something -- the figure layer's, a
record's, param's, the runner's -- are those layers' own, shown as they are (§1). A chart's
or a number's code name
(`sigma-y-norm`, `y1_rmse`) is not a word the page shows as a title: it stays the
file `deepc draw` writes and a link's address key. No word is on a hover but a greyed control's
reason and a chart's own -- a point's run and value, and the chart toolbar's button names (Zoom,
Pan, Zoom in, Zoom out, Reset axes), plotly's -- (Benchlab's `docs/design.md` decision 5, since
Benchlab 0.3.0); a reason is a sentence: it begins with a capital and keeps its articles where a
sentence needs them.

**Your word, 2026-11-11:** inputs (manipulated variables) are u1…, outputs (measurements) y1…,
true states x1…, time t — "the names everywhere (axis labels, chart buttons, EVERYWHERE)" — and
no units anywhere. Every piece follows it since 2026-11-17, when the Dataset piece and the old
Closed loop, which kept the plant class's names (e_x1, F1, …) and units, were deleted; the
reference is r1…, beside its output.

## 1. The page's own words

In `devkit/harness/words.toml`, grouped as the page is: the top line, the runs, the compare
table, the chart list, the cards, the form, what Run refuses in a setting, the runs in
progress, and what a click says. Beside them the page shows words it does not write itself:

- the figure layer's, on the cards: a greyed control's reason when it is the figure layer's
  (The chart declares no statistic · The runs draw different traces · …), a card that cannot
  draw saying why in its foot (e.g. First declare no panel 'cost'; drawn without them, the
  comparison would leave them out unsaid), and the legends: a run alone is its name; a trace
  every run holds identically: a, b — r1 (identical); a pooled line: Mean of 2 runs · y1
- a record's: the constants it keeps (each `group · key` as the record holds it, e.g.
  deepc_config · lambda_g), and why a file cannot be read, in the loader's words
- param's and the runner's, on a failed run in the strip and in the form's refusal, e.g.
  Integer parameter 'Config.N' must be at least 1, not 0.
- the controls a card has (the developer's cut, 2026-11-11): a chart card one, its runs
  (Overlay · Split · Mean · RMS); a number card two, X, offering the settings that differ
  between the shown runs by their label, and its runs (All · Mean · RMS)

## 2. The pieces' own words

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
- **Poles**
  - `pole-map`: Pole map · Re · Im · (no legend: one set of crosses, the poles of A)

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

### Closed loop (`closed_loop`)

Settings, by group -- `code name`: label (· its rows · add button) (no units):

- **Data**
  - `open_loop_run`: Open loop run
  - `x0`: x_0
  - `seed`: Seed
- **Reference**
  - `r`: r · rows r1 =, r2 =, … · + Reference
  - `duration`: Duration
- **Controller**
  - `Tini`: T_ini
  - `N`: N
  - `Q`: Q
  - `R`: R
  - `lambda_g`: λ_g
  - `lambda_y`: λ_y
  - `reg_norm_g`: Norm on g
  - `reg_norm_sigma_y`: Norm on σ_y

Charts of Closed loop (default run on the default open-loop run, the two masses), by group --
`code name`: title · bottom axis · ordinate · legend:

- **Outputs**
  - `y1`: y1 · t · y1 · y1, r1 (and `y2`…`y4` alike)
- **Inputs**
  - `u1`: u1 · t · u1 · u1, Lower bound of u1, Upper bound of u1 (and `u2`)
- **States**
  - `x1`: x1 · t · x1 · x1 (and `x2`…`x4`)
- **Disturbances**
  - `d1`: d1 · t · d1 · d1 (and `d2`)
- **Solver**
  - `solve-time`: Solve time · t · Solve time · Solve time
  - `cost`: J · t · J · J
  - `sigma-y-norm`: ‖σ_y‖₂ · t · ‖σ_y‖₂ · ‖σ_y‖₂

Numbers of Closed loop, by group -- `code name`: label:

- **Outputs**
  - `y1_rmse`: y1 RMSE (and `y2_rmse`…`y4_rmse`)
- **Inputs**
  - `clipped`: Clipped inputs
- **Solver**
  - `solves_optimal`: Solves: optimal (each status its own, e.g. `solves_optimal_inaccurate`: Solves: optimal inaccurate)
  - `solves_by_OSQP`: Solves by OSQP (`solves_by_SCS`: Solves by SCS)
  - `solve_time_mean`: Mean solve time
  - `solve_time_std`: Solve time std
  - `solve_time_max`: Max solve time
  - `sigma_y_norm_mean`: Mean ‖σ_y‖₂
  - `sigma_y_norm_max`: Max ‖σ_y‖₂
