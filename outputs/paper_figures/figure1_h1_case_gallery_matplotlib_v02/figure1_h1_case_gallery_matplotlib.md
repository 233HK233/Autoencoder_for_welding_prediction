# Figure 1 h1 case gallery matplotlib v02

This document explains what the current `Figure 1` version represents and how
you should interpret it in the context of the `horizon=1` three-class
forecasting task. It corresponds to the following files in the same directory:

- `figure1_h1_case_gallery_matplotlib.pdf`
- `figure1_h1_case_gallery_matplotlib.png`
- `figure1_h1_case_gallery_matplotlib.svg`

## What this figure represents

This figure is a qualitative case gallery for the `horizon=1` weld-seam state
forecasting task. Its purpose is not to report aggregate performance. Instead,
it shows representative local temporal cases across different seams and
different transition contexts so that you can visually inspect how the model
behaves around core regions and state-transition boundaries.

The figure is designed to answer three questions:

- Does the model remain reliable in stable core regions?
- Does the model become more uncertain near transition boundaries?
- Are local prediction patterns visually consistent across seams?

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

Each panel is therefore one seam-context pair. For example, `b01 |
1->2-boundary` shows a local temporal window from seam `b01` near the
transition from `S1` to `S2`.

## What each panel shows

Each panel visualizes one local time window centered around the selected
forecast target point.

The panel contains the following visual components:

- Three signal traces:
  - These traces are labeled as `High-Var Channel 1`, `High-Var Channel 2`,
    and `High-Var Channel 3`.
  - They correspond to the three highest-variance channels selected from the
    source seam.
- Colored background bands:
  - Blue indicates `S0 (Quasistable)`.
  - Yellow indicates `S1 (Nonstationary)`.
  - Red indicates `S2 (Instability)`.
- A red dashed vertical line:
  - This marks the prediction target point.
- A small arrow annotation labeled `Horizon=1`:
  - This emphasizes that the model predicts a future step rather than the
    current step.
- A prediction badge in the upper-right corner:
  - `Accurate` means the prediction matches the ground truth.
  - If a panel were incorrect, the figure would show the misclassified target
    state and the confidence.

## Axes and units

This `v02` version fixes the most important axis issue from the previous
iteration.

- The X axis is now a relative physical time axis.
- The target point is mapped to `t = 0`.
- The sampling period is assumed to be `0.01 s`, which corresponds to `100 Hz`.
- The displayed X axis therefore has the label `Relative Time (s)`.

This means you can now read each panel as a local temporal segment around the
forecast target in physically meaningful time units rather than raw array
indices.

The Y axis is still shown as `Normalized amplitude`. This version therefore
supports qualitative waveform comparison and relative shape inspection, but it
does not yet serve as a strict physical-amplitude comparison figure.

## What the current v02 content specifically shows

The current panel selection comes from the following high-scoring `horizon=1`
run:

- `outputs/teacher_h1_scan98_gpu1_v2/runs/`
  `single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep60_lr0.0003_bs128_k3_l3_d0.08_lat64_wd0.0002_seed14`

The panel selection file is:

- `outputs/teacher_h1_scan98_gpu1_v2/runs/.../figure1_panel_candidates.csv`

This `v02` figure contains no empty subplot. All `16` panel slots are filled.

Most panels are direct selections from the nominal seam. One panel is filled by
the agreed cross-seam backfill strategy:

- `c01 | S0-core` is backfilled from seam `c02`

This backfill avoids an empty panel while preserving the fixed `4 x 4` story
layout.

## What changed relative to the earlier version

Compared with the earlier matplotlib draft, this `v02` version introduces four
important corrections:

- It removes the hidden-empty-axis fallback and relies on a fully populated
  panel table.
- It replaces the raw index-like time axis with `Relative Time (s)`.
- It replaces generic legend names such as `Channel 0` with
  `High-Var Channel 1/2/3`.
- It removes the muddy gray future overlay and replaces it with a clean dashed
  target line plus `Horizon=1` arrow annotation.

## How you should describe this figure in the paper

You can describe the figure as a qualitative visualization of representative
forecasting behavior under seam-specific and transition-specific contexts.

A suitable English description is:

“Figure 1 presents a qualitative case gallery for the `horizon=1` state
forecasting task. Each row corresponds to one weld seam, and each column
corresponds to one representative temporal context, including core-state
regions and transition-boundary regions. The background state bands, relative
time axis, target-time marker, and `Horizon=1` annotation together illustrate
how the model behaves when forecasting future states from historical signal
observations.”

## Current limitations of this v02 version

This `v02` version resolves the most obvious presentation flaws raised during
review, but two limitations still remain if you want to make it fully
publication-ready for a strict top-tier submission:

- The signal traces still use high-variance channel labels instead of explicit
  physical channel names.
- The Y axis still uses normalized amplitude rather than explicit physical
  units such as voltage, current, or another domain-specific quantity.

These are presentation limitations, not structural flaws in the figure layout.

## Next steps

If you continue refining this figure, the most valuable next improvements are:

1. Replace `High-Var Channel 1/2/3` with true physical feature names.
2. Replace normalized amplitude with physically meaningful Y-axis units if the
   data semantics permit this conversion.
3. Tune font sizing, badge style, and legend density for the exact journal
   template you will submit to.
