# Figure 3 latent transfer and attention implementation spec

This document defines the implementation work needed to draw the proposed
paper-facing Figure 3, "Latent transfer and attention interpretability." It is
written as an executable specification for another coding agent. The figure
must provide visual and quantitative evidence for attention-guided latent
distillation, using the existing horizon-1 weld seam classification task.

The current repository already contains most checkpoints and prediction
outputs needed for inference. The implementation must add a compatible
attention export path, create a focused three-model representation analysis,
compute alignment metrics, and render a `3 x 3` figure.

## Bottom line

The figure is feasible, but the repository cannot draw the strict final
version without additional implementation. The main missing pieces are an
official attention return interface, a three-model embedding export, a
dimension-compatible Student-only latent representation, a unified
UMAP/PaCMAP projection, and source tables for quantitative alignment metrics.

The implementation must not silently reuse the existing
`outputs/representation_analysis/h1/` artifacts as final Figure 3 evidence.
Those artifacts are useful references, but they target a different `2 x 3`
model-comparison point-cloud figure and use per-model t-SNE projection.

## Figure purpose

Figure 3 must answer three reviewer questions that are not fully answered by
aggregate classification metrics.

- Does the SEAL-Weld student move toward the teacher's latent structure?
- Is the improvement measurable beyond a subjective two-dimensional plot?
- Does the attention mechanism focus on the same temporal boundary evidence
  after distillation?

The figure must use the same `horizon=1` task and the same test sample order
for every model. Do not mix `horizon=0` and `horizon=1` results.

## Required figure layout

The final figure must use a `3 x 3` layout. Each row has a different evidence
role, and every panel must have a source table under the output directory.

| Panel | Title | Required content |
| --- | --- | --- |
| `(a)` | `Teacher latent` | Unified UMAP or PaCMAP projection for teacher `z`. |
| `(b)` | `Student-only latent` | Same projection coordinates for Student-only `z`. |
| `(c)` | `SEAL-Weld Student latent` | Same projection coordinates for Full SEAL-Weld `z`. |
| `(d)` | `Class-centroid cosine` | Teacher-to-student centroid cosine by `S0`, `S1`, and `S2`. |
| `(e)` | `Class-wise linear CKA` | Teacher-to-student linear CKA by class. |
| `(f)` | `Boundary-distance similarity` | Teacher-to-student similarity by `Far`, `Mid`, and `Near`. |
| `(g)` | `Teacher attention` | Boundary-window attention or saliency heatmap. |
| `(h)` | `Student-only attention` | Same windows and time axis as panel `(g)`. |
| `(i)` | `SEAL-Weld Student attention` | Same windows and time axis as panel `(g)`. |

Use one shared class color mapping in the first row:

- `S0`: blue, `#1f77b4`
- `S1`: orange, `#ff7f0e`
- `S2`: green, `#2ca02c`

## Existing code and artifact status

The implementation must start from the current repository state, not from an
assumed clean experiment environment. The following files and behaviors are
already present.

| Item | Status | Source |
| --- | --- | --- |
| H1 dataset | Available | `Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz` |
| Teacher checkpoint | Available | `outputs/teacher_h1_scan98_gpu1_v2/runs/...seed14/best_single_tcn.pth` |
| Full SEAL-Weld checkpoint | Available | `ablation_experiments/h1/results/figure2_matched_distillation_ablation/seed14/full_seal_weld/.../best_student_distill.pth` |
| Student-only checkpoint | Available | `ablation_experiments/h1/results/figure2_matched_distillation_ablation/seed14/student_only_anchor/.../best_student_only.pth` |
| Existing model `z` output | Available | `AttentionTCNClassifier.forward()` returns `(logits, z)`. |
| Official attention output | Missing | `AttentionTCNClassifier.forward()` currently calls attention with `need_weights=False`. |
| Boundary bin logic | Available | `build_figure4_temporal_case_results.py` defines `Near`, `Mid`, and `Far`. |
| Figure 4 attention workaround | Available | `run_attention_inference()` manually calls `model.attn(... need_weights=True)`. |
| UMAP/PaCMAP dependency | Missing in current environment | `umap` and `pacmap` are not installed. |

Use `outputs/paper_figures/figure2_distillation_ablation_v02/source_tables/component_ablation_metrics.csv`
as the source of truth for the matched Figure 2 anchor runs.

## Model selection

The final Figure 3 must compare exactly three models. The method names and
run sources must remain stable across all panels.

| Display name | Role | Checkpoint file | Notes |
| --- | --- | --- | --- |
| `Teacher` | 18D upper-bound teacher | `best_single_tcn.pth` | Uses all 18 input features. |
| `Student-only` | 13D supervised student | `best_student_only.pth` | Uses deployable 13D input only. |
| `SEAL-Weld Student` | 13D distilled student | `best_student_distill.pth` | Uses deployable 13D input only. |

The implementation must save the resolved run directory and checkpoint path
for every model in the manifest. Relative paths in `run_args.json` must be
resolved from the repository root.

## Blocking dimension decision

The first-row unified projection requires every latent vector to have the same
feature dimension before concatenation. The current matched anchors do not all
meet that requirement.

- `Teacher` uses `AttentionTCNClassifier` with `channels=64`, so `z` is
  `128` dimensions because it concatenates mean and max pooled features.
- `SEAL-Weld Student` uses the matched full run with `channels=64`, so `z` is
  also `128` dimensions.
- `Student-only` from the matched anchor uses the ablation-1 builder, where
  `parse_channels(..., latent_dim=80)` forces the final channel count to `80`,
  so `z` is `160` dimensions.

For the final paper figure, use the `strict_raw_latent` route:

1. Add a `--student-latent-dim` or equivalent option to
   `ablation_experiments/scripts/train_ablation1_student_only.py`.
2. Use that option in `parse_channels()` instead of the hard-coded
   `latent_dim=80`.
3. Train a matched Student-only anchor with `student_latent_dim=64`, which
   produces a `128`-dimensional `z`.
4. Record the new run in the Figure 3 manifest.

For a draft-only figure, the agent may implement an `aligned_latent` route by
mapping Student-only `z` into a common dimension. If it uses this route, it
must write `projection_space=aligned_latent` into the manifest and the figure
caption must state that the first row uses a post-hoc aligned latent space.
Do not present post-hoc alignment as raw latent comparison.

## Required code changes

The implementation must keep all existing training scripts compatible. Existing
calls that expect `(logits, z)` must continue to work without modification.

### Add attention output to `AttentionTCNClassifier`

Update `models_tcn.py` so the attention model supports an inference-only
attention export path.

The required interface is:

```python
def forward(
    self,
    x: torch.Tensor,
    return_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

When `return_attention=False`, return exactly `(logits, z)`. When
`return_attention=True`, call `self.attn()` with:

```python
need_weights=True
average_attn_weights=False
```

Then return `(logits, z, attention_weights)`.

The expected attention shape from PyTorch is `[batch, heads, target_step,
source_step]` when `batch_first=True` and `average_attn_weights=False`.

### Update Figure 4 helper usage

After adding the official attention output path, update
`build_figure4_temporal_case_results.py` to call:

```python
logits, z, attn_weights = model(x_batch, return_attention=True)
```

Do not keep duplicate manual attention logic unless it is needed as a fallback.
The goal is to have one canonical model interface.

### Add focused Figure 3 export script

Create a new script:

```text
export_figure3_latent_attention.py
```

This script must load the three selected models, run inference on the same H1
test set, export embeddings and attention data, and compute source metrics.

The script must support these arguments:

```text
--dataset-npz
--output-root
--projection-method
--device
--batch-size
--random-state
--student-only-run-dir
--seal-weld-run-dir
--teacher-run-dir
```

Use safe defaults from this document, but make every run directory overridable.

### Add focused Figure 3 renderer

Create a new script:

```text
render_figure3_latent_attention.py
```

This script must read only source tables from the Figure 3 output directory.
It must not load checkpoints or recompute inference. This separation keeps
data generation and visual rendering testable.

## Required output layout

All new outputs must go into a new versioned directory. Do not overwrite the
existing Figure 3 comparison artifacts.

Use this directory layout:

```text
outputs/paper_figures/figure3_latent_attention_v01/
  figure3_latent_attention.png
  figure3_latent_attention.pdf
  figure3_latent_attention.svg
  manifest.csv
  qa_notes.txt
  embeddings/
    teacher_embeddings_test.npz
    student_only_embeddings_test.npz
    seal_weld_student_embeddings_test.npz
  source_tables/
    selected_model_runs.csv
    unified_projection_coordinates.csv
    class_centroid_cosine_similarity.csv
    linear_cka_by_class.csv
    similarity_by_boundary_bin.csv
    selected_boundary_windows.csv
    boundary_attention_heatmap.csv
    reproduced_classification_metrics.csv
```

The export script must create the output directory. If `v01` exists, create the
next available version, such as `v02`.

## Embedding file contract

Each embedding file must be a `.npz` file with a stable set of arrays. The
arrays must preserve the H1 test-set sample order.

| Field | Shape | Required | Description |
| --- | --- | --- | --- |
| `Z_test` | `[356, latent_dim]` | Yes | Classifier-input latent vector. |
| `logits` | `[356, 3]` | Yes | Raw logits for `S0`, `S1`, and `S2`. |
| `prob` | `[356, 3]` | Yes | Softmax probabilities. |
| `y_true` | `[356]` | Yes | Ground-truth H1 labels. |
| `y_pred` | `[356]` | Yes | Predicted labels from `argmax(logits)`. |
| `sample_index` | `[356]` | Yes | Index in `X_test_full`. |
| `seam_id` | `[356]` | Yes | Numeric seam id from the dataset. |
| `seam_name` | `[356]` | Yes | Seam name from `seam_name_order`. |
| `start_idx` | `[356]` | Yes | Window start index. |
| `target_idx` | `[356]` | Yes | H1 target index. |
| `attention_weights` | `[356, heads, 5, 5]` | Yes for attention models | Raw self-attention weights. |
| `attention_ribbon` | `[356, 5]` | Yes for attention models | Reduced and normalized source-step attention. |

The `attention_ribbon` must be computed by averaging
`attention_weights` over heads and target steps, then normalizing each sample
so its five time-step weights sum to `1.0`.

## Unified projection protocol

The first row must use one projection model fitted on the concatenated latent
matrix from the three methods. Do not fit one projection per panel.

Use this protocol for `strict_raw_latent`:

1. Load `Z_test` for `Teacher`, `Student-only`, and `SEAL-Weld Student`.
2. Confirm all three arrays have the same second dimension.
3. Standardize each model's `Z_test` with a separate `StandardScaler`.
4. Concatenate the standardized arrays along the sample axis.
5. Fit one UMAP or PaCMAP reducer on the concatenated array.
6. Split the resulting coordinates back by model.
7. Save all coordinates to `unified_projection_coordinates.csv`.

The coordinate table must include these columns:

| Column | Description |
| --- | --- |
| `model_name` | `Teacher`, `Student-only`, or `SEAL-Weld Student`. |
| `sample_index` | Test-set sample index. |
| `x_2d` | First projection coordinate. |
| `y_2d` | Second projection coordinate. |
| `y_true` | Ground-truth class id. |
| `y_pred` | Predicted class id for the same model. |
| `correct` | Whether `y_pred == y_true`. |
| `seam_name` | Human-readable seam name. |
| `start_idx` | Window start index. |
| `target_idx` | H1 target index. |
| `projection_method` | `umap`, `pacmap`, or explicit fallback. |
| `projection_space` | `strict_raw_latent` or `aligned_latent`. |
| `projection_random_state` | Projection random seed. |

If neither `umap-learn` nor `pacmap` is installed, the script must stop with a
clear error for the final figure. A PCA fallback is acceptable only when the
user explicitly requests a draft.

## Alignment metric protocol

The second row must quantify teacher-student representation alignment. Compute
metrics on standardized high-dimensional latent vectors, not on the
two-dimensional projection.

### Class-centroid cosine

For every class `c`, compute the centroid for each model:

```text
mu_model,c = mean(Z_model[y_true == c], axis=0)
```

Then compute cosine similarity between the teacher centroid and each student
centroid:

```text
cosine(mu_teacher,c, mu_student,c)
```

This metric requires matching latent dimensions. In `strict_raw_latent`, the
dimensions must match naturally. In `aligned_latent`, compute centroid cosine
only after the documented alignment transform.

Save one row per `(student_model, class_id)` to
`class_centroid_cosine_similarity.csv`.

Required columns:

```text
student_model,class_id,class_label,n_samples,cosine_similarity,
teacher_latent_dim,student_latent_dim,projection_space
```

### Linear CKA by class

Linear CKA can compare matrices with different feature dimensions, so it is
valid for both `strict_raw_latent` and the current unmatched Student-only
anchor.

For matrices `X` and `Y` with the same rows:

```text
Xc = X - mean(X, axis=0)
Yc = Y - mean(Y, axis=0)
cka = ||Xc.T @ Yc||_F^2 / (||Xc.T @ Xc||_F * ||Yc.T @ Yc||_F)
```

Compute one CKA value per class and per student model. Save the results to
`linear_cka_by_class.csv`.

Required columns:

```text
student_model,class_id,class_label,n_samples,linear_cka,
teacher_latent_dim,student_latent_dim
```

### Boundary-distance similarity

Reuse the boundary-distance logic from `build_figure4_temporal_case_results.py`.
The binning rule is:

- `Near`: nearest boundary distance `<= 5`
- `Mid`: nearest boundary distance `6` to `15`
- `Far`: nearest boundary distance `> 15`

For each bin and student model, compute at least these values:

- mean paired sample cosine if latent dimensions match or an aligned space is
  used
- linear CKA within the bin
- mean squared distance to teacher if dimensions match or an aligned space is
  used
- number of samples

Save the results to `similarity_by_boundary_bin.csv`.

Required columns:

```text
student_model,boundary_distance_bin,n_samples,linear_cka,
mean_paired_cosine,mean_teacher_student_mse,projection_space
```

## Boundary-window and attention protocol

The third row must use the same windows for all three models. Do not select
different windows per model.

Build `selected_boundary_windows.csv` from the H1 test set with these rules:

1. Load raw labels for each seam from `Data/raw_data`.
2. Compute the `0->1` and `1->2` transition index for each seam.
3. Annotate every H1 test sample with its nearest transition and distance.
4. Select only `0->1` and `1->2` boundary windows.
5. Prefer `Near` windows where Teacher and SEAL-Weld are both correct.
6. Include a balanced number of `0->1` and `1->2` windows when possible.
7. Keep the selected sample list identical across Teacher, Student-only, and
   SEAL-Weld Student.

For a compact paper figure, use one of these display modes:

- `mean_attention`: average attention ribbon across all selected boundary
  windows and draw one row per boundary type.
- `sample_heatmap`: draw a heatmap with one row per selected sample.

Use `sample_heatmap` if the selected set has at most `32` windows. Use
`mean_attention` if the selected set is larger.

The `boundary_attention_heatmap.csv` table must include these columns:

```text
model_name,method_slug,sample_index,seam_name,boundary_type,
boundary_distance_bin,start_idx,target_idx,input_step,relative_step,
relative_time_s,attention_weight,y_true,y_pred,correct
```

If the implementation uses gradient saliency instead of attention, it must
rename the column to `temporal_importance`, write `importance_method=saliency`
in the manifest, and use panel titles that say `saliency`, not `attention`.

## Rendering requirements

The renderer must make the figure readable at paper scale. It must avoid
chart effects that make quantitative interpretation harder.

Use these visual rules:

- Use a landscape canvas suitable for a full-width paper figure.
- Use panel labels `(a)` through `(i)`.
- Keep a shared legend for `S0`, `S1`, and `S2` in the first row.
- Use identical x-axis and y-axis limits for panels `(a)` through `(c)`.
- Use fixed y-axis ranges where possible for panels `(d)` through `(f)`.
- Use the same color scale for panels `(g)` through `(i)`.
- Export `png`, `pdf`, and `svg`.
- Save all source tables used by the renderer.

The renderer must not recompute embeddings or attention. It must fail if any
required source table is missing.

## Validation checks

The export script must run validation before it writes the final manifest. A
failed validation must stop the run and write a short `qa_notes.txt`.

Run these checks:

1. Confirm the dataset path resolves to
   `Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`.
2. Confirm every embedding file has `356` rows.
3. Confirm every `y_true` array exactly matches the dataset `y_test`.
4. Confirm `sample_index`, `seam_id`, `start_idx`, and `target_idx` match
   across all three models.
5. Confirm probability rows sum to `1.0` within `1e-5`.
6. Confirm `Z_test` and attention arrays contain no NaN or infinite values.
7. Confirm the attention ribbon sums to `1.0` for every sample.
8. Confirm the projection was fitted once on the concatenated latent matrix.
9. Confirm `projection_space=strict_raw_latent` for the final paper version.
10. Confirm reproduced test metrics match each source run's
    `evaluation_metrics.txt` within `0.1` percentage points.
11. Confirm the renderer exports `png`, `pdf`, and `svg`.

Write validation results to `qa_notes.txt`, even when all checks pass.

## Test requirements

Add focused tests for helper functions and file contracts. The tests must not
run full training.

Recommended tests:

- `tests/test_figure3_latent_attention_export.py`
- `tests/test_render_figure3_latent_attention.py`

The tests must cover:

- `AttentionTCNClassifier(x)` still returns exactly `(logits, z)`.
- `AttentionTCNClassifier(x, return_attention=True)` returns
  `(logits, z, attention_weights)`.
- attention weights reduce to a normalized five-step ribbon.
- the unified projection helper rejects mismatched latent dimensions in
  `strict_raw_latent` mode.
- linear CKA returns `1.0` for identical matrices within tolerance.
- boundary-distance binning returns `Near`, `Mid`, and `Far` correctly.
- the renderer fails with a clear error when a required source table is
  missing.

Run at least these commands before declaring the implementation complete:

```bash
python -m pytest tests/test_figure3_latent_attention_export.py
python -m pytest tests/test_render_figure3_latent_attention.py
python -m pytest tests/test_build_figure4_temporal_case_results.py
```

If the implementation changes `models_tcn.py`, also run the existing training
interface tests that cover TCN and distillation behavior.

## Suggested implementation sequence

Use this sequence to keep the work reviewable. Each step produces a concrete
artifact before moving to the next step.

1. Add the compatible `return_attention` interface to `AttentionTCNClassifier`.
2. Add tests that prove the default return value remains `(logits, z)`.
3. Refactor Figure 4 attention inference to use the new interface.
4. Add a `student_latent_dim` option to the Student-only training script.
5. Train or register a `z=128` Student-only anchor for final Figure 3.
6. Implement the Figure 3 export script and write embedding `.npz` files.
7. Implement unified UMAP or PaCMAP projection on concatenated embeddings.
8. Implement centroid cosine, class-wise CKA, and boundary-bin similarity.
9. Implement boundary-window selection and attention heatmap export.
10. Implement the renderer from source tables only.
11. Run validation and tests.
12. Write `qa_notes.txt` and update the manifest.

## Out of scope

This task must stay focused on Figure 3. Do not use it to change unrelated
model training behavior or replace existing paper figures.

The following work is out of scope:

- changing the H1 dataset split
- mixing H0 and H1 outputs
- rerunning baseline model comparisons
- replacing the existing ROC Figure 3 comparison
- changing class definitions for `S0`, `S1`, or `S2`
- hand-editing metrics in source tables

## Caption requirements

The final figure caption must make the evidence logic clear. The first
sentence must state the conclusion.

Use this caption structure:

1. State that latent distillation moves the deployable student toward the
   teacher's representation and attention pattern near state boundaries.
2. Explain that panels `(a)` through `(c)` use one projection fitted on the
   concatenated latent matrix.
3. Explain that panels `(d)` through `(f)` quantify teacher-student alignment
   in the original latent space or the declared aligned space.
4. Explain that panels `(g)` through `(i)` use the same selected boundary
   windows for all models.
5. State the dataset, horizon, and label mapping.

If `aligned_latent` is used, the caption must say so explicitly. Do not bury
that note in the manifest only.

## Next steps

Give this document to the implementation agent as the source of truth. The
agent must first decide whether to produce the final `strict_raw_latent` paper
version or a draft `aligned_latent` version. After that decision, the rest of
the work is deterministic: update the attention interface, export the three
model artifacts, compute source tables, render the figure, and run the
validation checks.
