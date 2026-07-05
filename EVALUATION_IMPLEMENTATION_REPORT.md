# Evaluation Implementation Report

## Phase 0.4 Official Metric Bridge

Adopted the official metric bridge direction from:

```text
https://github.com/royerlab/kaggle-cell-tracking-competition
```

Official package details recorded:

- package name: `tracking-cellmot`
- import namespace: `tracking_cellmot`
- official metric entry points: `tracking_cellmot.metrics.evaluate`, `evaluate_datasets`, `per_sample_metrics`, `summarise`
- official dataset loader: `tracking_cellmot.io.open_dataset`
- graph format: tracksdata GEFF

No modeling work has been started.

## Git Checkpoint Status

The workspace was not a valid Git repository. An empty broken `.git` placeholder existed and blocked `git init`; after verifying it was empty, it was removed and `git init` succeeded.

The required Phase 0.3 checkpoint commit could not be created because the sandbox denies writes to `.git/index` and `.git/config`:

```text
fatal: Unable to create '.git/index.lock': Permission denied
error: could not lock config file .git/config: Permission denied
```

## Official Submission Bridge

Added:

- `src/evaluation/official_bridge.py`
- `submission_df_to_tracksdata(df)`
- `tracksdata_to_submission_df(graphs)`

The bridge enforces the exact official schema:

```text
id, dataset, row_type, node_id, t, z, y, x, source_id, target_id
```

Rules preserved:

- identity is dataset-scoped: `(dataset, node_id)`
- `node_id` may repeat across datasets but not within one dataset
- `id` is a row id only and is regenerated as `0..N-1`
- node rows carry `t,z,y,x` and use `source_id=target_id=-1`
- edge rows carry `source_id,target_id` and use `node_id=t=z=y=x=-1`
- sample submission row count is not forced

Round-trip tests cover multi-dataset graphs, duplicate `node_id` values across datasets, and a division.

## Official-Style Metric API

Added:

- `src/evaluation/official_metric.py`
- `evaluate_one_dataset(pred_graph, gt_graph, scale, t_true=None)`
- `evaluate_datasets_official_style(pairs, scale, t_true_map=None)`
- `adjusted_edge_jaccard(edge_jaccard, t_pred, t_true)`
- `read_estimated_number_of_nodes(geff_path)`

Implemented adjusted Edge Jaccard formula:

```text
adjusted_jaccard = max(
    0,
    edge_jaccard * (1 - 0.1 * (T_pred - T_true) / T_true)
)
```

`T_pred` is total predicted nodes.

`T_true` source for training GEFF:

```text
extra.estimated_number_of_nodes
```

read from GEFF metadata in `zarr.json` or `.zattrs` when present.

If `T_true` is unavailable:

- raw edge/division metrics may still be returned
- `adjusted_edge_jaccard` is `NaN`
- `final_score` is `NaN`
- a warning is logged

## Fallback Metric Status

The dependency-light fallback implements the documented sparse-GT edge behavior:

- TP: predicted edge whose matched endpoints are connected by a GT edge
- FN: GT edge not recovered
- FP: predicted edge that conflicts with annotated GT connectivity by target parent or source child
- other predicted edges are ignored because GT is sparse

The fallback division score is not the official tolerant `tracking_cellmot` division metric. It is exact out-degree division matching for local smoke tests only.

The old Phase 0.2/0.3 approximation functions remain for backward compatibility but are deprecated:

- `node_overprediction_penalty_approx`
- `edge_jaccard_approx`

They now emit/propagate deprecation warnings and are not described as official metrics.

## Official Dependencies

Optional dependency notes were added to `requirements-dev.txt`.

Install attempts:

- `tracksdata` installed into `.official_deps` with `--no-deps`
- `tracking-cellmot` built and installed into `.official_deps` with `--no-deps`

Import status:

```text
tracking_cellmot/tracksdata import failed because tracksdata requires missing dependency `geff`
```

So guarded optional imports remain in use, and official cross-check tests skip cleanly in this local environment.

## Official Smoke Test

Added:

```text
python -m src.evaluation.official_smoke_test --train-root /kaggle/input/competitions/biohub-cell-tracking-during-development/train --limit 5
```

It is designed to:

- use `tracking_cellmot.io.open_dataset` when available
- avoid full image loading when supported
- read tracks and scale
- read `estimated_number_of_nodes`
- self-evaluate each training graph
- print nodes, edges, divisions, adjusted edge score, division score, and final score

Local result:

```text
no GEFF stores found under \kaggle\input\competitions\biohub-cell-tracking-during-development\train
```

## GEFF Decoder Status

GEFF node mapping remains:

- node ids: `nodes/ids`
- node properties: `nodes/props`
- required properties: `t`, `z`, `y`, `x`
- structured props are decoded by field names
- positional props require metadata names

GEFF edge mapping:

- source/target pairs from `edges/ids`

Division extraction:

- divisions are inferred as exactly two outgoing child edges unless official division metadata is later discovered

## Tests

Command:

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath '.pytest_deps').Path
& 'C:\Users\hp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/evaluation -q
```

Result:

```text
39 passed, 1 skipped, 4 warnings in 3.51s
```

Official cross-check:

```text
skipped: tracking_cellmot unavailable/import-incomplete locally
```

## Remaining Open Questions

- Whether `T_true` is exposed for test data or only inside the hidden Kaggle scorer.
- Whether this local environment can install `tracking-cellmot` with all heavy dependencies safely.
- Whether a full official cross-check passes once `tracking_cellmot`, `tracksdata`, and `geff` dependencies are available.
- Official tolerant division metric behavior is bridged only when the official package is available; fallback division scoring is not official.
