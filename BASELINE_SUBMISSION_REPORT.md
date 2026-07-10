# Phase 1.1 Baseline Submission Report

## Design

Phase 1.1 adds a lightweight CPU-only baseline submission pipeline at:

```bash
python -m src.baseline.make_baseline_submission
```

The pipeline writes the exact official mixed node/edge CSV schema:

```text
id, dataset, row_type, node_id, t, z, y, x, source_id, target_id
```

It regenerates `id` as `0..N-1`, keeps `node_id` local to each dataset, emits official sentinel values for node and edge rows, and validates the output with the existing local validator when `--sample-submission` is provided.

## Modes

`--mode smoke` discovers `*.zarr` stores under `--test-root`, inspects image shape metadata or local fixture arrays safely, creates a tiny deterministic valid graph for each dataset, writes `submission.csv`, and validates it. This mode is for end-to-end plumbing only.

`--mode classical` runs a simple per-frame detector and one-to-one linker. It detects peaks in voxel space, writes voxel coordinates, and links only adjacent frames using physical distances.

`--mode eval-on-train` runs the same classical detector on train Zarr stores that have paired GEFF annotations, then evaluates with the existing official-style metric helper. If train data or optional official dependencies are unavailable, it skips cleanly.

Eval-on-train now supports the real Kaggle GEFF node-property layout observed with `zarr 3.2.1`:

```text
edges/ids | shape=(50, 2) | dtype=uint64
nodes/ids | shape=(52,) | dtype=uint64
nodes/props/t/values | shape=(52,) | dtype=int64
nodes/props/z/values | shape=(52,) | dtype=int64
nodes/props/y/values | shape=(52,) | dtype=int64
nodes/props/x/values | shape=(52,) | dtype=int64
```

The previous blocker was an assumption that `nodes/props` was directly readable as an array/table. The decoder now keeps that legacy path, but falls back to the official grouped property arrays above.

## Classical Strategy

For each dataset, the baseline:

- detects OME-Zarr multiscale metadata when available and selects full-resolution level `0`
- verifies the image array rank is `(T,Z,Y,X)`
- processes one timepoint at a time
- normalizes each frame using robust percentile clipping `[p1,p99]`
- optionally smooths with `scipy.ndimage.gaussian_filter`
- ranks local peak candidates by intensity
- keeps at most `--max-detections-per-frame`
- links detections from `t` to `t+1` only
- uses one-to-one assignment through `scipy.optimize.linear_sum_assignment` when available
- gates links by `--link-max-um`
- predicts no divisions in Phase 1.1

Local tests use small `.npy` fixture arrays inside `.zarr` directories so they do not require Kaggle data or the optional `zarr` package.

## Defaults

```text
threshold_percentile=99.5
min_distance_xy=5
min_distance_z=2
max_detections_per_frame=300
sigma=1.0
link_max_um=5.0
scale_zyx_um=(1.625, 0.40625, 0.40625)
```

These defaults follow the Oracle findings: prioritize detection recall and valid one-to-one linking first. Coordinate precision beyond roughly `1-2 um` is less urgent than missing nodes or wrong links, and division prediction is deferred because division removal had a smaller targeted cost than node dropout or edge swaps.

## Known Limitations

This is not a competitive model yet. It has no learned detector, no segmentation, no motion model beyond adjacent-frame assignment, no division prediction, and only basic intensity-peak detection. Dense or noisy frames may require parameter tuning, especially `threshold_percentile`, `sigma`, and `max_detections_per_frame`.

## Local Usage

```bash
python -m src.baseline.make_baseline_submission \
  --test-root path/to/test \
  --sample-submission path/to/sample_submission.csv \
  --output submission.csv \
  --mode smoke
```

## Kaggle Usage

```bash
python -m src.baseline.make_baseline_submission \
  --test-root /kaggle/input/competitions/biohub-cell-tracking-during-development/test \
  --sample-submission /kaggle/input/competitions/biohub-cell-tracking-during-development/sample_submission.csv \
  --output /kaggle/working/submission.csv \
  --mode classical \
  --threshold-percentile 99.5 \
  --max-detections-per-frame 300 \
  --sigma 1.0 \
  --link-max-um 5.0
```

## Eval-On-Train Status

The CLI supports `--mode eval-on-train`, but this local workspace does not contain Kaggle train Zarr/GEFF data. If `tracking_cellmot` is unavailable, eval-on-train decodes paired train GEFF graphs and then skips official scoring with a clear warning.

Run it inside Kaggle with:

```bash
python -m src.baseline.make_baseline_submission \
  --train-root /kaggle/input/competitions/biohub-cell-tracking-during-development/train \
  --mode eval-on-train \
  --limit 5
```

## Test Status

Baseline tests currently pass locally on synthetic fixtures. Full-suite output is recorded in the final task response.

Verified Kaggle Phase 1.1 status before this fix:

- smoke mode succeeded on real test data
- classical mode `--limit 1` succeeded on real test Zarr
- `44b6_0113de3b` resolved `array=0`, `shape=(100,64,256,256)`
- submission validation passed
- with `--max-detections-per-frame 150`, the limit-1 output contained `15000` nodes and `9085` edges
