# Figure 1 h1 case gallery matplotlib v03

This document explains what the current `Figure 1` version represents and how
you should interpret it in the context of the `horizon=1` three-class
forecasting task. It corresponds to the following files in the same directory:

- `figure1_h1_case_gallery_matplotlib.pdf`
- `figure1_h1_case_gallery_matplotlib.png`
- `figure1_h1_case_gallery_matplotlib.svg`

## What this figure represents

This figure is a qualitative case gallery for the `horizon=1` weld-seam state
forecasting task. Its purpose is not to summarize global accuracy. Instead, it
shows representative local signal windows under different seam-specific and
transition-specific contexts, and it now explicitly visualizes how the model
prediction trajectory compares with the ground-truth trajectory in each panel.

The figure is designed to answer three questions:

- Does the model remain reliable in stable core regions?
- Does the model become more uncertain near transition boundaries?
- How does the predicted state trajectory differ from the true state trajectory
  in local temporal neighborhoods?

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

Each panel is therefore one seam-context pair. For example, `a01 |
1->2-boundary` shows a local temporal window from seam `a01` near the
transition from `S1` to `S2`.

## What each panel shows

Each panel visualizes one local time window centered around the selected
forecast target point.

The panel contains the following visual components:

- Three signal traces:
  - These traces are labeled as `High-Var Channel 1`, `High-Var Channel 2`,
    and `High-Var Channel 3`.
  - They correspond to the three automatically selected highest-variance
    channels from the source seam.
- Colored background bands:
  - Blue indicates `S0 (Quasistable)`.
  - Yellow indicates `S1 (Nonstationary)`.
  - Red indicates `S2 (Instability)`.
- A red dashed vertical line:
  - This marks the prediction target point.
- A small arrow annotation labeled `Horizon=1`:
  - This emphasizes that the model predicts one future step rather than the
    current step.
- A small inset in the upper-right corner:
  - This inset shows two local state trajectories as step curves.
  - The black step line is `GT`.
  - The red dashed step line is `Pred`.
  - This inset makes the local prediction behavior more transparent than a
    plain text badge because it directly shows where the prediction follows or
    deviates from the true state evolution.

## Axes and units

This `v03` version keeps the relative physical time axis introduced in `v02`.

- The X axis is a relative physical time axis.
- The target point is mapped to `t = 0`.
- The sampling period is assumed to be `0.01 s`, which corresponds to `100 Hz`.
- The displayed X axis therefore has the label `Relative Time (s)`.

This means you can interpret each panel as a local temporal segment around the
forecast target in physically meaningful time units rather than raw array
indices.

The Y axis is still shown as `Normalized amplitude`. This version therefore
supports qualitative waveform comparison and local shape inspection, but it is
not yet intended for strict cross-panel physical-amplitude comparison.

## What the current v03 content specifically shows

The current panel selection comes from the following high-scoring `horizon=1`
run:

- `outputs/teacher_h1_scan98_gpu1_v2/runs/`
  `single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep60_lr0.0003_bs128_k3_l3_d0.08_lat64_wd0.0002_seed14`

The panel selection file is:

- `outputs/teacher_h1_scan98_gpu1_v2/runs/.../figure1_panel_candidates.csv`

This `v03` figure contains no empty subplot. All `16` panel slots are filled.

Most panels are direct selections from the nominal seam. One panel is filled by
the agreed cross-seam backfill strategy:

- `c01 | S0-core` is backfilled from seam `c02`

This backfill avoids an empty panel while preserving the fixed `4 x 4` layout.

## What changed relative to the earlier versions

Compared with the earlier matplotlib versions, this `v03` version introduces a
more explicit representation of the model prediction result:

- It retains the fully populated panel layout from `v02`.
- It retains the relative physical time axis `Relative Time (s)`.
- It retains the `High-Var Channel 1/2/3` legend naming.
- It retains the clean `Horizon=1` target marker without the muddy gray future
  overlay.
- It replaces the earlier text-like correctness cue with a local `GT vs Pred`
  state-trajectory inset, which is more direct and more scientifically
  interpretable.

## How you should describe this figure in the paper

You can describe the figure as a qualitative visualization of local signal
patterns and local state-trajectory agreement between prediction and ground
truth under seam-specific and transition-specific contexts.

A suitable English description is:

“Figure 1 presents a qualitative case gallery for the `horizon=1` state
forecasting task. Each row corresponds to one weld seam, and each column
corresponds to one representative temporal context, including core-state
regions and transition-boundary regions. In each panel, the local waveform, the
background state segmentation, the target-time marker, and the inset `GT vs
Pred` step trajectories jointly illustrate how the model forecasts future
states from historical observations.”

## Current limitations of this v03 version

This `v03` version is stronger than the earlier drafts because it makes the
prediction behavior more explicit, but two limitations still remain if you want
to push it closer to a strict top-tier submission standard:

- The signal traces still use high-variance channel labels rather than explicit
  physical feature names.
- The Y axis still uses normalized amplitude rather than explicit physical
  units such as voltage, current, or another domain-specific quantity.

These are presentation limitations, not structural flaws in the figure itself.

## Next steps

If you continue refining this figure, the most valuable next improvements are:

1. Replace `High-Var Channel 1/2/3` with true physical feature names.
2. Replace normalized amplitude with physically meaningful Y-axis units if the
   data semantics permit this conversion.
3. Simplify the inset styling further if you want an even cleaner journal-ready
   visual hierarchy.
