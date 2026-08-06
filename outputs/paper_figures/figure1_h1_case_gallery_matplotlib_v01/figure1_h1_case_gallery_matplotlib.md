# Figure 1 h1 case gallery matplotlib v01

This document explains what the current `Figure 1` version represents and how
you should read it in the context of the `horizon=1` three-class forecasting
task. It corresponds to the following files in the same directory:

- `figure1_h1_case_gallery_matplotlib.pdf`
- `figure1_h1_case_gallery_matplotlib.png`
- `figure1_h1_case_gallery_matplotlib.svg`

## What this figure represents

This figure is a qualitative case gallery for the `horizon=1` weld-seam state
forecasting task. Its purpose is not to summarize overall accuracy. Instead,
it shows how the model behaves on representative local temporal regions across
different seams and different state-transition contexts.

The figure answers three questions:

- Does the model remain stable in core regions of a given state?
- Does the model become more uncertain near state-transition boundaries?
- Do local prediction patterns remain consistent across seams?

## How the layout is organized

This version uses a `4 x 4` matrix layout.

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
0->1-boundary` shows a local temporal window from seam `a01` around the
transition from `S0` to `S1`.

## What each panel shows

Each panel visualizes one local time window centered around the selected
`target_idx`.

The panel contains the following visual elements:

- Three signal traces:
  - These are the three automatically selected high-variance channels from the
    source seam.
  - They show the local waveform pattern seen around the prediction target.
- Colored background bands:
  - Blue background indicates `S0 (Quasistable)`.
  - Yellow background indicates `S1 (Nonstationary)`.
  - Red background indicates `S2 (Instability)`.
- A dark dashed vertical line:
  - This marks the forecast target position.
- A gray shaded region to the right of the dashed line:
  - This marks the future-side region used to visually emphasize the
    `horizon=1` prediction setting.
- A prediction badge in the upper-right corner:
  - `Accurate` means `Pred == GT`.
  - If a panel were wrong, the figure would show `Misclassified as Sx` together
    with the confidence.

## Axes and units

The panel X axis is currently shown as `Relative time step`. At this stage, it
is still expressed in sample-step space rather than physical seconds.

The panel Y axis is currently shown as `Normalized amplitude`. This means the
current version emphasizes qualitative pattern comparison rather than direct
physical amplitude comparison across panels.

You should interpret this figure as a qualitative, publication-oriented signal
pattern visualization, not as a figure for exact physical-value reading.

## What the current v01 content specifically shows

The current panel selection comes from the high-scoring run:

- `outputs/teacher_h1_scan98_gpu1_v2/runs/`
  `single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep60_lr0.0003_bs128_k3_l3_d0.08_lat64_wd0.0002_seed14`

The panel selection file is:

- `outputs/teacher_h1_scan98_gpu1_v2/runs/.../figure1_panel_candidates.csv`

For this `v01` figure:

- Most panels are direct selections from the target seam.
- One panel is not available under the strict original seam-context rule:
  - `c01 | S0-core`
- To avoid an empty panel, this slot is backfilled from another seam under the
  agreed policy of cross-seam filling:
  - `c01 | S0-core` is backfilled from seam `c02`

This means the figure is fully populated with real signal data, but one panel
does not originate from its nominal row seam.

## How you should describe this figure in the paper

You can describe the figure as a qualitative visualization of representative
forecasting cases across seam-specific and transition-specific contexts.

A suitable English description is:

“Figure 1 presents a qualitative case gallery for the `horizon=1` state
forecasting task. Each row corresponds to one weld seam, and each column
corresponds to one representative temporal context, including core-state
regions and transition-boundary regions. The colored state bands, target-time
marker, and future-region shading jointly illustrate how the model behaves when
forecasting future states from historical observations.”

## Current limitations of this v01 version

This `v01` figure is materially stronger than the previous custom-canvas PNG,
but it still has two limitations if you want to make it fully publication-ready
for a strict top-tier submission:

- The signal traces still use generic labels `Channel 0`, `Channel 1`, and
  `Channel 2`, rather than explicit physical channel names.
- The axes still use normalized amplitude and relative time step, rather than
  physical units such as seconds, voltage, or current.

These are presentation limitations, not logical errors in the figure.

## Next steps

If you continue refining this figure, the most important next improvements are:

1. Replace generic channel labels with actual physical channel names.
2. Convert the X axis from sample steps to physical time units.
3. Replace normalized amplitude with physically meaningful units if the channel
   semantics support that conversion.
