# Evaluation Implementation Report

## Phase 0.5 Official Runtime Verification

The official Kaggle runtime is now considered verified from a real Kaggle Notebook run.

Verified data status:

- Kaggle root exists: `True`
- train root exists: `True`
- test root exists: `True`
- train GEFF stores: `199`
- train Zarr stores: `199`
- test Zarr stores: `4`
- test GEFF stores: none

Verified optional package status inside Kaggle after isolated install:

- `geff`: OK
- `tracksdata`: OK
- `tracking_cellmot`: OK
- `tracking_cellmot.metrics`: OK
- `tracking_cellmot.io`: OK

Previous local import root cause:

- `tracksdata` pulled NumPy 2.5.
- `numba` failed because it required NumPy 2.4 or less.
- Verified fix: install optional official dependencies in an isolated target with `numpy<2.5`.
- Working versions observed: `numpy 2.4.6`, `numba 0.66.0`.

No modeling work has been started.

## Sparse-GT Caveat

The official labels are extremely sparse. In the verified example, dataset `44b6_0113de3b` has only `52` annotated GT nodes while `estimated_number_of_nodes` is `25755`, roughly `0.2%` annotation coverage.

Self-evaluation scores above `1.0` are a ceiling artifact of evaluating `pred == GT` against sparse annotations with `T_pred << T_true`. They are not a modeling progress signal. Real submissions predict on full images and will score far lower.

The metric ignores unmatched predictions because GT is sparse. False positives are effectively free unless they conflict with annotated GT connectivity. The only broad over-prediction cost is the flat `T_pred` penalty. Wrong associations near annotated GT cells are costly because they create both an FP edge and an FN edge.

## Official Submission Schema

Official columns:

```text
id, dataset, row_type, node_id, t, z, y, x, source_id, target_id
```

Identity rules:

- `id` is a row id only and is regenerated as `0..N-1`.
- `node_id` is local to each `dataset`.
- the cell key is `(dataset, node_id)`.
- sample submission row count is not forced.

## Official Dataset Loader

Verified call convention:

```python
open_dataset(
    TRAIN / dataset_name,
    normalize=False,
    require_tracks=True,
    load_image=False,
)
```

Example verified on `44b6_0113de3b`:

- scale: `(1.625, 0.40625, 0.40625)`
- image_shape: `(100, 64, 256, 256)`
- zarr_path: `/kaggle/input/competitions/biohub-cell-tracking-during-development/train/44b6_0113de3b.zarr`
- image is `None`: `True`
- tracks type: `tracksdata.graph._rustworkx_graph.IndexedRXGraph`
- nodes: `52`
- edges: `50`

## T_true Metadata Path

Patched `read_estimated_number_of_nodes(geff_path)` to read the verified official path:

```python
attrs["geff"]["extra"]["estimated_number_of_nodes"]
```

It supports both:

- Zarr v3 `zarr.json`, using the `attributes` object
- Zarr v2 `.zattrs`

Legacy fallback:

```python
attrs["extra"]["estimated_number_of_nodes"]
```

is still supported with a `DeprecationWarning`.

If `estimated_number_of_nodes` is absent, the helper returns `None` and emits a warning. For test data, `T_true` is unavailable at inference because test has no `.geff`; it lives in the hidden scorer.

## Official Metric Formula

Edge Jaccard:

```text
edge_jaccard = TP / (TP + FP + FN)
```

Adjusted Edge Jaccard:

```text
adjusted_jaccard = max(
    0,
    edge_jaccard * (1 - 0.1 * (T_pred - T_true) / T_true)
)
```

Final score:

```text
final_score = adjusted_edge_jaccard + 0.1 * division_jaccard
```

If no divisions exist in a split, official behavior drops the division term.

## Verified Real Self-Evaluation

Five train GEFF self-evaluations verified in Kaggle:

| dataset | nodes | edges | T_true | edge_tp/fp/fn | division_tp/fp/fn | adj_edge_jaccard |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `44b6_0113de3b` | 52 | 50 | 25755 | 50/0/0 | 0/0/0 | 1.0997980974568045 |
| `44b6_0b24845f` | 51 | 49 | 32795 | 49/0/0 | 0/0/0 | 1.099844488489099 |
| `44b6_0c582fdc` | 71 | 70 | 27958 | 70/0/0 | 0/0/0 | 1.0997460476428929 |
| `44b6_0db75fae` | 157 | 151 | 15335 | 151/0/0 | 0/0/0 | 1.0989761982393218 |
| `44b6_12dfb391` | 788 | 773 | 58672 | 773/0/0 | 1/0/0 | 1.0986569402781565 |

Official `summarise(rows)` verified:

```json
{
  "n": 5,
  "edge_jaccard": 1.0,
  "division_jaccard": 1.0,
  "division_tp": 1,
  "division_fp": 0,
  "division_fn": 0,
  "node_recall": 1.0,
  "adj_edge_jaccard": 1.0988762387126818,
  "n_adj": 5,
  "score": 1.198876238712682
}
```

These values are locked in an optional Kaggle-runtime regression test that skips cleanly when Kaggle data or official packages are unavailable.

## Bridge And Fallback Status

Added official bridge modules:

- `src/evaluation/official_bridge.py`
- `src/evaluation/official_metric.py`
- `src/evaluation/official_smoke_test.py`

The fallback metric remains clearly marked as fallback. It is not called official unless cross-checked through `tracking_cellmot`.

Deprecated old approximations:

- `node_overprediction_penalty_approx`
- `edge_jaccard_approx`

They remain only for backward compatibility.

## Tests

Command:

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath '.pytest_deps').Path
& 'C:\Users\hp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/evaluation -q
```

Latest local result:

```text
47 passed, 2 skipped, 4 warnings in 1.45s
```

## Phase 1.0 Oracle Framework

Added:

- `src/oracle/oracle_experiments.py`
- `tests/oracle/test_oracle_perturbations.py`
- `ORACLE_FINDINGS.md`

The oracle framework perturbs canonical official submission DataFrames and rebuilds graphs through the submission bridge. This keeps dataset-scoped identity and official sentinel values intact.

Implemented experiments:

- `node_dropout`
- `coord_jitter`
- `node_count_adjustment_curve`
- `division_removal`
- `edge_swap`

CLI:

```bash
python -m src.oracle.oracle_experiments \
  --train-root /kaggle/input/competitions/biohub-cell-tracking-during-development/train \
  --limit 5 \
  --seed 42 \
  --output-dir /kaggle/working/oracle
```

Optional:

```bash
--prefer-divisions --limit 20
```

Local status:

```text
train root does not exist: \kaggle\input\competitions\biohub-cell-tracking-during-development\train
```

So official integration skipped locally and no oracle CSVs were created here. In Kaggle, the framework uses `open_dataset(..., normalize=False, require_tracks=True, load_image=False)` and official `tracking_cellmot` metrics when available. Local tests are non-official dependency-light checks only.

## Remaining Open Questions

- Whether `T_true` is exposed for test data anywhere outside the hidden scorer.
- Whether this local Windows environment can install the complete official stack without dependency conflicts.
- Official division tolerance behavior is available through `tracking_cellmot`; local fallback division scoring remains non-official.
