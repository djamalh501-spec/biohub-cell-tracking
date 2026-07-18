# Phase 1.1 Baseline and Phase 1.2A Adaptive Detection Report

## Immutable Reference Baseline

- Kaggle Version 7 is immutable.
- Public leaderboard score: `0.180`
- Local official 15-dataset score: `0.1800343922572182`
- Reference parameters:
  - `detection_policy=fixed`
  - `threshold_percentile=99.7`
  - `sigma=1.0`
  - `max_detections_per_frame=300`
  - `link_max_um=5.0`

The local official score agrees with the public leaderboard score at leaderboard precision. Version 7 remains the fixed comparison point for later phases and must not be replaced by adaptive experiments.

## Phase 1.1 Design

The Phase 1.1 pipeline is a deterministic, CPU-only classical baseline implemented in `src/baseline/make_baseline_submission.py`. It discovers datasets dynamically from `*.zarr` directories, reads one `(Z,Y,X)` timepoint at a time, and never loads a complete image volume into memory.

The fixed detector performs robust percentile normalization, optional Gaussian smoothing, intensity ranking, and spatial suppression. It retains at most `max_detections_per_frame` candidates. Adjacent frames are linked with deterministic one-to-one assignment in physical units, gated by `link_max_um`. This phase does not predict divisions.

Generated submissions use the official mixed node/edge schema:

```text
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
```

Row IDs are regenerated as `0..N-1`, node IDs remain local to each dataset, edge endpoints must exist in the same dataset, and output is checked by the existing official-schema validator when `--sample-submission` is supplied.

Eval-on-train uses the official `tracking_cellmot` scoring path when its optional dependencies are available. It loads ground-truth tracks without image data, converts predictions through the official bridge, and reports node recall, Edge Jaccard, adjusted Edge Jaccard, predicted and estimated true node counts, their ratio, and the summary score. The local fallback is used only when official packages are unavailable and is labeled `FALLBACK`.

## Phase 1.2A Adaptive Detection

Phase 1.2A adds the opt-in `adaptive-count` policy while leaving the fixed detector unchanged. Each frame uses the existing normalization, smoothing, intensity ranking, and spatial suppression to produce candidate scores. The robust score threshold is:

```text
robust_threshold = median(scores) + adaptive_mad_k * 1.4826 * MAD(scores)
```

The number of finite scores strictly above this threshold is clamped between `adaptive_min_detections` and `adaptive_max_detections`, subject to candidate availability. The strongest candidates are retained. A causal rolling median smooths frame-level targets using only the current and previous frames, preserving streaming execution.

The policy handles empty candidate sets, zero MAD, low-contrast frames, non-finite scores, insufficient candidates, and incomplete initial smoothing windows. For every adaptive dataset it reports total `T_pred`, detections-per-frame mean/min/max, raw and smoothed target means, and frame counts that hit the lower and upper bounds.

The implementation is deterministic for a fixed seed and uses only the existing lightweight NumPy, SciPy, Pandas, and Zarr stack.

## Defaults

The CLI keeps `--detection-policy fixed` as the default so Phase 1.1 behavior remains the reference path. The compatibility default for `--threshold-percentile` remains `99.5`; all immutable-reference and Phase 1.2A Kaggle commands below pass `99.7` explicitly.

Adaptive option defaults are:

```text
adaptive_mad_k=3.0
adaptive_min_detections=80
adaptive_max_detections=400
adaptive_count_smoothing_window=5
```

The shared defaults used by the documented adaptive runs are `sigma=1.0`, `min_distance_xy=5`, `min_distance_z=2`, `link_max_um=5.0`, and `seed=42`.

## Known Limitations

This remains a classical intensity-peak baseline. It has no learned detector, segmentation model, division predictor, or motion model beyond adjacent-frame assignment. Adaptive-count changes how many spatially suppressed candidates are retained; it does not improve localization quality or association semantics. Dense, noisy, and very low-contrast frames can still require parameter tuning.

Local verification uses synthetic Zarr fixtures because the official Kaggle image and GEFF stores are not present in this workspace. Real-data scores and runtime must therefore be measured in Kaggle.

## Kaggle Commands

### Fixed Baseline Submission

This reproduces the immutable Version 7 fixed-policy configuration:

```bash
python -m src.baseline.make_baseline_submission \
  --test-root /kaggle/input/competitions/biohub-cell-tracking-during-development/test \
  --sample-submission /kaggle/input/competitions/biohub-cell-tracking-during-development/sample_submission.csv \
  --output /kaggle/working/submission.csv \
  --mode classical \
  --detection-policy fixed \
  --threshold-percentile 99.7 \
  --max-detections-per-frame 300 \
  --sigma 1.0 \
  --link-max-um 5.0
```

### Adaptive Eval-on-Train

This evaluates one adaptive configuration on 15 dynamically discovered training datasets:

```bash
python -m src.baseline.make_baseline_submission \
  --train-root /kaggle/input/competitions/biohub-cell-tracking-during-development/train \
  --mode eval-on-train \
  --limit 15 \
  --detection-policy adaptive-count \
  --threshold-percentile 99.7 \
  --sigma 1.0 \
  --adaptive-mad-k 3.0 \
  --adaptive-min-detections 80 \
  --adaptive-max-detections 400 \
  --adaptive-count-smoothing-window 5 \
  --link-max-um 5.0
```

### Fast Adaptive Sweep

This runs the recommended 12-configuration grid on three dynamically discovered training datasets and writes results after every configuration:

```bash
python -m src.baseline.sweep_adaptive \
  --train-root /kaggle/input/competitions/biohub-cell-tracking-during-development/train \
  --limit 3 \
  --threshold-percentile 99.7 \
  --sigma 1.0 \
  --min-distance-xy 5 \
  --min-distance-z 2 \
  --adaptive-min-detections 80 \
  --adaptive-count-smoothing-window 5 \
  --adaptive-mad-k-values 2.5,3.0,3.5 \
  --adaptive-max-detections-values 300,400 \
  --link-max-um-values 5.0,7.0 \
  --seed 42 \
  --output /kaggle/working/adaptive_sweep_fast.csv
```

### Final Selected-Configuration Evaluation

The two `--config` tuples below are placeholders in the accepted `MAD_K,MAX_DETECTIONS,LINK_MAX_UM` format. Replace them with the two fast-sweep winners before running the 15-dataset evaluation.

```bash
python -m src.baseline.sweep_adaptive \
  --train-root /kaggle/input/competitions/biohub-cell-tracking-during-development/train \
  --limit 15 \
  --threshold-percentile 99.7 \
  --sigma 1.0 \
  --min-distance-xy 5 \
  --min-distance-z 2 \
  --adaptive-min-detections 80 \
  --adaptive-count-smoothing-window 5 \
  --seed 42 \
  --config 2.5,400,5.0 \
  --config 3.0,400,7.0 \
  --output /kaggle/working/adaptive_sweep_final.csv
```

### Adaptive Hidden-Test Submission

This creates a validated hidden-test submission for one selected adaptive configuration:

```bash
python -m src.baseline.make_baseline_submission \
  --test-root /kaggle/input/competitions/biohub-cell-tracking-during-development/test \
  --sample-submission /kaggle/input/competitions/biohub-cell-tracking-during-development/sample_submission.csv \
  --output /kaggle/working/submission.csv \
  --mode classical \
  --detection-policy adaptive-count \
  --threshold-percentile 99.7 \
  --sigma 1.0 \
  --adaptive-mad-k 3.0 \
  --adaptive-min-detections 80 \
  --adaptive-max-detections 400 \
  --adaptive-count-smoothing-window 5 \
  --link-max-um 5.0
```

## Verification

Pre-commit verification covers the complete pytest suite, both CLI help surfaces, Markdown structure, command integrity, and `git diff --check`. Kaggle-only integration tests skip cleanly when official data or optional runtime packages are unavailable.

No packaging ZIP is created by this phase, and no changes should be committed until review is approved.
