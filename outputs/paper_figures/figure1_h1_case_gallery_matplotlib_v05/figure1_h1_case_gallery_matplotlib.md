# Figure 1 h1 case gallery matplotlib v05

This document explains what the current `Figure 1` version represents and how
you should interpret it in the context of the `horizon=1` three-class
forecasting task. It corresponds to the following files in the same directory:

- `figure1_h1_case_gallery_matplotlib.pdf`
- `figure1_h1_case_gallery_matplotlib.png`
- `figure1_h1_case_gallery_matplotlib.svg`

## What this figure represents

This figure is a qualitative case gallery for the `horizon=1` weld-seam state
forecasting task. Its role is to show representative local temporal regions
under different seam-specific and transition-specific contexts, while also
making the prediction behavior visually explicit through a local `GT vs Pred`
state-evolution inset.

It is designed to answer three questions:

- Does the model remain reliable in stable core regions?
- Does the model become more uncertain near transition boundaries?
- How closely does the predicted local state trajectory follow the true local
  state trajectory?

## How the layout is organized

This version uses a fixed `4 x 4` matrix layout.

- Rows correspond to seams:
  - `a01`
  - `b01`
  - `c01`
  - `c02`
- Columns correspond to temporal contexts:
  - `S0-core`
  - `0->1-boundary`
  - `S1-core`
  - `1->2-boundary`

Each panel therefore represents one seam-context pair. For example,
`c02 | 1->2-boundary` shows a local temporal window from seam `c02` near the
transition from `S1` to `S2`.

## What each panel shows

Each panel visualizes one local time window centered around the selected
forecast target point.

The panel contains the following visual elements:

- Three signal traces:
  - These are labeled as `1st Principal High-Var Channel`,
    `2nd Principal High-Var Channel`, and
    `3rd Principal High-Var Channel`.
  - They correspond to the three highest-variance channels selected from the
    source seam.
- Colored background bands:
  - Blue indicates `S0 (Quasistable)`.
  - Yellow indicates `S1 (Nonstationary)`.
  - Red indicates `S2 (Instability)`.
- A dark target marker:
  - A low-contrast dashed vertical line marks the prediction target point.
  - A small arrow and `Horizon=1` text indicate that the task predicts the
    next future step.
- A local prediction-result label in the upper-left corner:
  - `Pred: Accurate` means the target prediction is correct.
  - If the target prediction is wrong, the label reports the predicted class
    and confidence.
- A compact inset in the upper-right corner:
  - The black step line is `GT`.
  - The red dashed step line is `Pred`.
  - This inset shows local state evolution around the target point.

## Axes and units

This version keeps the physically meaningful relative time axis introduced in
the earlier revision.

- The X axis is a relative physical time axis.
- The forecast target is mapped to `t = 0`.
- The sampling period is assumed to be `0.01 s`, which corresponds to `100 Hz`.
- The X-axis label is therefore `Relative Time (s)`.

This means you can interpret each panel as a local physical-time neighborhood
around the forecast target rather than as a raw index slice.

The Y axis is still shown as `Normalized amplitude`. This supports qualitative
signal-shape comparison, but it is not yet a strict physical-amplitude figure.

## What the current v05 content specifically shows

The panel selection comes from the following high-scoring `horizon=1` run:

- `outputs/teacher_h1_scan98_gpu1_v2/runs/`
  `single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep60_lr0.0003_bs128_k3_l3_d0.08_lat64_wd0.0002_seed14`

The panel selection file is:

- `outputs/teacher_h1_scan98_gpu1_v2/runs/.../figure1_panel_candidates.csv`

This `v05` figure contains no empty subplot. All `16` panel slots are filled.

Most panels are direct selections from the nominal seam. One panel is filled by
the agreed cross-seam backfill strategy:

- `c01 | S0-core` is backfilled from seam `c02`

This preserves the fixed `4 x 4` layout without leaving visual holes.

## What changed relative to the earlier versions

Compared with `v04`, this `v05` version is primarily a visual-density cleanup.

The main updates are:

- The `GT vs Pred` inset is smaller, so it occludes less of the raw waveform.
- The inset trajectories are thicker, so they remain readable despite the
  reduced size.
- The prediction-result label is now rendered as plain bold text instead of a
  boxed badge, which reduces decorative clutter.
- The `Horizon=1` target marker is now low-contrast dark gray or black rather
  than strong red, so it behaves as a structural cue rather than a competing
  visual highlight.
- The top and right spines are removed from each subplot, producing a cleaner,
  more open journal-style layout.

## How you should describe this figure in the paper

You can describe this figure as a qualitative visualization of local waveform
patterns and local state-evolution agreement between prediction and ground
truth under seam-specific and transition-specific contexts.

A suitable English description is:

“Figure 1 presents a qualitative case gallery for the `horizon=1` state
forecasting task. Each row corresponds to one weld seam, and each column
corresponds to one representative temporal context, including core-state
regions and transition-boundary regions. In each panel, the raw waveform, the
background state segmentation, the relative-time target marker, the local
prediction label, and the compact `GT vs Pred` inset jointly illustrate how
the model forecasts future states from historical observations.”

## Current limitations of this v05 version

This `v05` version is visually cleaner and more publication-oriented than the
earlier drafts, but two limitations still remain if you want to push it closer
to a final top-tier submission figure:

- The signal traces still use abstract high-variance channel names rather than
  true physical feature names.
- The Y axis still uses normalized amplitude rather than explicit physical
  units such as voltage, current, or another domain-specific quantity.

These are remaining presentation limitations, not logical flaws in the figure
construction.

## Next steps

If you continue refining this figure, the most valuable next improvements are:

1. Replace the abstract high-variance channel names with true physical feature
   names.
2. Replace normalized amplitude with physically meaningful Y-axis units if the
   signal semantics permit that conversion.
3. Perform one last round of journal-specific typography tuning if needed.
