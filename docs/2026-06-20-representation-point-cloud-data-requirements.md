# H1 representation point-cloud data requirements

This document defines the supplemental results needed to draw a point-cloud
representation figure for the `horizon=1` comparison experiment. The goal is
to give another coding agent one executable specification for inference
outputs, feature extraction, projection, metrics, and validation.

The point-cloud figure is a qualitative representation analysis. It must align
with the existing Figure 3 quantitative comparison, rather than replace the
classification metrics already shown in Figure 3.

## Scope

This figure belongs to the method comparison results. It visualizes whether
different model architectures learn more separable latent representations on
the same one-step-ahead test set.

- Figure type: experimental results
- Figure role: qualitative representation visualization
- Primary experiment: model comparison
- Task: one-step-ahead state prediction
- Dataset:
  `Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`
- Label set: `S0`, `S1`, and `S2`
- Recommended method order:
  `Teacher`, `Student`, `GRU`, `LSTM`, `Inception`, `Transformer`

Do not use this figure as the main ablation evidence. Ablations are better
served by metric tables, bars, box plots, or `mean +/- std` summaries. An
ablation point-cloud figure can be added later as supplementary analysis, but
it is not the main target of this task.

## Existing quantitative anchor

The new point-cloud figure must use the same task and method set as the
current Figure 3. This keeps the qualitative embedding visualization tied to
the existing quantitative results.

Use these existing files as the source of truth for method selection and
metric alignment:

- Figure 3 rendered output:
  `outputs/paper_figures/fig3_h1_comparison_layout_v03/figure3_h1_comparison.png`
- Figure 3 panel values:
  `outputs/figure 3/figure_exports/figure3_h1_comparison_panel_values.csv`
- Figure 3 comparison source table:
  `outputs/figure 3/source_tables/comparison_summary.csv`

The coding agent must read `comparison_summary.csv` and use its rows as the
comparison registry. Do not manually select a different set of runs unless a
human explicitly changes the figure scope.

## Required supplemental results

The required new results come from inference, not from retraining. For every
selected model, load the trained checkpoint, run the same H1 test set once,
and save the intermediate representation immediately before the final
classifier.

Each model must produce one embedding file with these arrays or equivalent
columns:

| Field | Shape | Required | Description |
| --- | --- | --- | --- |
| `Z_test` | `[n_test, latent_dim]` | Yes | Classifier-input latent representation for each test sample. |
| `logits` | `[n_test, 3]` | Yes | Raw class logits from the same forward pass. |
| `prob` | `[n_test, 3]` | Yes | Softmax probabilities for `S0`, `S1`, and `S2`. |
| `y_true` | `[n_test]` | Yes | Ground-truth labels copied from the H1 test set. |
| `y_pred` | `[n_test]` | Yes | Predicted labels from `argmax(prob)`. |
| `sample_index` | `[n_test]` | Yes | Row index in `X_test_full`. |
| `seam_id` | `[n_test]` | Recommended | Numeric seam identifier if present in the dataset. |
| `seam_name` | `[n_test]` | Recommended | Human-readable seam name, such as `a01`. |
| `start_idx` | `[n_test]` | Recommended | Window start index if present in the dataset. |
| `target_idx` | `[n_test]` | Recommended | H1 target index if present in the dataset. |
| `model_name` | scalar or metadata | Yes | Display name used in the figure. |
| `run_path` | scalar or metadata | Yes | Source run directory. |
| `checkpoint_path` | scalar or metadata | Yes | Checkpoint used for inference. |
| `dataset_npz` | scalar or metadata | Yes | Dataset used for inference. |
| `feature_layer` | scalar or metadata | Yes | Name of the extracted representation layer. |

The sample order must match `X_test_full` in
`Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`. If a model uses a
feature subset, such as the deploy student, the agent must keep the same test
sample order after feature selection.

## Feature extraction rules

Each model must export the vector that feeds the final classifier head. This
keeps the embedding definition comparable across architectures.

Use the following extraction targets:

| Method | Representation to save |
| --- | --- |
| `Teacher` | The `z` returned by the TCN-Attention teacher before its classifier. |
| `Student` | The `z` returned by the deploy student before its classifier. |
| `GRU` | The final pooled or projected hidden vector before the classifier. |
| `LSTM` | The feature vector after `fc1` and activation, before `fc2`. |
| `Inception` | The pooled `z` vector before its classifier. |
| `Transformer` | The pooled encoder representation before the classifier. |

If a baseline model does not currently return this representation, add an
inference-only helper such as `forward_features()` or a safe optional return
path. Do not change training behavior.

## Recommended output layout

Write all new artifacts under a single output directory. This keeps the point
cloud analysis separate from training runs and existing paper figures.

Recommended directory:

```text
outputs/representation_analysis/h1/
  manifest.csv
  embeddings/
    teacher_embeddings_test.npz
    student_embeddings_test.npz
    gru_embeddings_test.npz
    lstm_embeddings_test.npz
    inception_embeddings_test.npz
    transformer_embeddings_test.npz
  projections/
    tsne_coordinates.csv
    umap_coordinates.csv
  metrics/
    representation_metrics.csv
    reproduced_classification_metrics.csv
  figures/
    h1_representation_tsne.png
    h1_representation_tsne.pdf
    h1_representation_tsne.svg
```

The implementation can omit UMAP if the environment does not have `umap-learn`.
If UMAP is omitted, record that decision in `manifest.csv`.

## Projection requirements

The point-cloud panels must be generated from `Z_test`. The projection method
must be deterministic and recorded in the output files.

Use this protocol unless a human changes it:

- Standardize `Z_test` per model before projection.
- Use t-SNE as the default projection.
- Use the same projection parameters for all methods.
- Set `random_state=42`.
- Use `init="pca"` when available.
- Use `learning_rate="auto"` when available.
- Use `perplexity=30` when valid for the sample count.
- If `n_test` is too small for `perplexity=30`, use
  `min(30, floor((n_test - 1) / 3))`.

Each method can be projected separately for its own panel. Do not compare the
absolute axis coordinates across panels, because t-SNE axes are arbitrary.
Compare visual cluster compactness, overlap, and class separation.

The combined coordinate CSV must include these columns:

| Column | Description |
| --- | --- |
| `model_name` | Figure display name. |
| `sample_index` | Test-set row index. |
| `x_2d` | First projected coordinate. |
| `y_2d` | Second projected coordinate. |
| `y_true` | Ground-truth class id. |
| `y_pred` | Predicted class id. |
| `correct` | Whether `y_pred == y_true`. |
| `seam_name` | Seam name if available. |
| `start_idx` | Window start index if available. |
| `target_idx` | H1 target index if available. |
| `projection_method` | `tsne` or `umap`. |
| `projection_random_state` | Random seed used by the projection. |

## Representation metrics

The point-cloud image is qualitative, so the agent must also compute a small
set of representation-space metrics. These metrics must be computed on the
standardized original latent vectors, not on the two-dimensional t-SNE or UMAP
coordinates.

Compute these metrics for each model:

| Metric | Direction | Description |
| --- | --- | --- |
| `silhouette_score` | Higher is better | Class separation based on true labels. |
| `davies_bouldin_index` | Lower is better | Cluster compactness and separation. |
| `calinski_harabasz_score` | Higher is better | Ratio of between-cluster to within-cluster dispersion. |
| `mean_intra_class_distance` | Lower is better | Average distance from samples to their class centroid. |
| `mean_inter_class_centroid_distance` | Higher is better | Average distance between class centroids. |
| `inter_intra_distance_ratio` | Higher is better | `mean_inter_class_centroid_distance / mean_intra_class_distance`. |

Save these metrics to:

```text
outputs/representation_analysis/h1/metrics/representation_metrics.csv
```

Include `n_test`, `latent_dim`, and `feature_layer` in the same table.

## Figure requirements

The main figure must use a compact small-multiple layout. It must be readable
as a supplement to the existing Figure 3 quantitative comparison.

Use this layout:

- Canvas: `2 x 3` panels
- Panel order:
  1. `Teacher`
  2. `Student`
  3. `GRU`
  4. `LSTM`
  5. `Inception`
  6. `Transformer`
- Color mapping:
  - `S0`: blue
  - `S1`: orange
  - `S2`: green
- Marker size: small enough to avoid hiding dense regions.
- Legend: one shared legend outside the panels.
- Axis labels: `Dimension 1` and `Dimension 2`.
- Panel titles: method names only.

Optional: Mark misclassified points with a thin outline or a second marker
style. If this makes the figure visually crowded, do not include the
misclassification overlay in the main version.

## Validation checks

The agent must run these checks before declaring the result complete. These
checks prevent a good-looking plot from using the wrong task or mismatched
sample order.

1. Confirm every embedding file has the same `n_test`.
2. Confirm every `y_true` array exactly matches the H1 dataset `y_test`.
3. Confirm every method uses
   `Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`.
4. Confirm `prob` rows sum to `1.0` within numerical tolerance.
5. Confirm there are no `NaN` or infinite values in `Z_test`.
6. Recompute test accuracy and macro-F1 from `y_true` and `y_pred`.
7. Compare recomputed metrics with Figure 3 source values and flag any
   difference greater than `0.1` percentage points.
8. Confirm the output figure exists in `png`, `pdf`, and `svg` formats.
9. Confirm `manifest.csv` records every run path, checkpoint path, dataset
   path, projection method, and random seed.

If any validation check fails, stop and write the failure into the manifest or
a short `qa_report.md` under `outputs/representation_analysis/h1/`.

## Out of scope

This task is intentionally limited to inference-time representation analysis.
Do not add unrelated experiments while implementing it.

The following items are out of scope:

- Retraining any model.
- Mixing `horizon=0` and `horizon=1` results.
- Replacing the current Figure 3 quantitative metrics.
- Using training-set embeddings as the main figure.
- Adding new model families not present in Figure 3.
- Changing the existing training scripts in a way that alters training
  behavior.

## Next steps

Use this sequence to implement the analysis cleanly.

1. Read `outputs/figure 3/source_tables/comparison_summary.csv`.
2. Resolve each selected run directory, checkpoint, and model configuration.
3. Load `Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`.
4. Run test-set inference for every selected model.
5. Save one embedding file per model.
6. Recompute classification metrics and validate them against Figure 3.
7. Compute representation metrics on original standardized `Z_test`.
8. Generate t-SNE coordinates and save the coordinate table.
9. Render the `2 x 3` point-cloud figure in `png`, `pdf`, and `svg`.
10. Write a manifest or QA report that records any assumptions, skipped
    artifacts, or metric mismatches.
