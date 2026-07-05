# Oracle Findings

Phase 1.0 adds a reproducible oracle experiment framework. No modeling, detector, GPU, Cellpose, 3D U-Net, ultrack, or image loading is used.

## Runtime Status

Local workspace result:

```text
train root does not exist: \kaggle\input\competitions\biohub-cell-tracking-during-development\train
```

Therefore no oracle CSVs were generated locally. In Kaggle, run:

```bash
python -m src.oracle.oracle_experiments \
  --train-root /kaggle/input/competitions/biohub-cell-tracking-during-development/train \
  --limit 5 \
  --seed 42 \
  --output-dir /kaggle/working/oracle
```

Optional division-focused run:

```bash
python -m src.oracle.oracle_experiments \
  --train-root /kaggle/input/competitions/biohub-cell-tracking-during-development/train \
  --limit 20 \
  --seed 42 \
  --output-dir /kaggle/working/oracle \
  --prefer-divisions
```

## Expected Outputs

- `node_dropout.csv`
- `coord_jitter.csv`
- `node_count_adjustment_curve.csv`
- `division_removal.csv`
- `edge_swap.csv`

Default local path:

```text
outputs/oracle/
```

Kaggle path:

```text
/kaggle/working/oracle/
```

## Experiments

`node_dropout` measures sensitivity to node recall by dropping node rows and all incident edges.

`coord_jitter` measures spatial error sensitivity by adding Gaussian physical-coordinate noise and converting back to voxel units.

`node_count_adjustment_curve` isolates the official `T_pred / T_true` adjustment without adding fake nodes.

`division_removal` removes one daughter edge from each two-child parent.

`edge_swap` rewires edges to incorrect but forward-time targets.

## Strategic Reading

Expected levers to rank after Kaggle run:

1. node recall
2. spatial accuracy near the 7 um threshold
3. edge correctness
4. node-count calibration through `T_pred`
5. divisions

Sparse-GT warning: self-evaluation scores above `1.0` are ceiling artifacts caused by `pred == GT` and tiny annotated graphs compared with `T_true`; they are not real model progress signals.
