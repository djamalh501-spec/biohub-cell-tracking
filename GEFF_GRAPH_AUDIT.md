# GEFF Graph Audit

## Verified Kaggle Runtime Status

- Kaggle root exists: `True`
- train GEFF stores: `199`
- train Zarr stores: `199`
- test Zarr stores: `4`
- official imports work after isolated install with `numpy<2.5`

Previous dependency issue:

```text
Numba needs NumPy 2.4 or less. Got NumPy 2.5.
```

Verified working versions:

- `numpy 2.4.6`
- `numba 0.66.0`

## Test Set

Known test dataset stems from `test/*.zarr` / sample submission:

| dataset | image_shape | has_geff |
| --- | --- | --- |
| `44b6_0113de3b` | unavailable in local workspace; audit script records this in Kaggle | no |
| `44b6_0b24845f` | unavailable in local workspace; audit script records this in Kaggle | no |
| `6bba_05b6850b` | unavailable in local workspace; audit script records this in Kaggle | no |
| `6bba_05db0fb1` | unavailable in local workspace; audit script records this in Kaggle | no |

The test set has no `.geff` annotations. Therefore `T_true` is unavailable at inference and lives in the hidden Kaggle scorer.

## Train GEFF Metadata

Correct `T_true` metadata path:

```python
attrs["geff"]["extra"]["estimated_number_of_nodes"]
```

Supported storage:

- Zarr v3: `zarr.json` attributes
- Zarr v2: `.zattrs`

Legacy fallback:

```python
attrs["extra"]["estimated_number_of_nodes"]
```

is deprecated and emits a warning.

## Decoder Mapping

- node ids: `nodes/ids`
- node properties: `nodes/props`
- required node properties: `t`, `z`, `y`, `x`
- named properties: decoded by structured dtype field names
- positional properties: decoded only when metadata names the property columns
- edge ids: `edges/ids` source/target pairs
- divisions: one source node with exactly two outgoing target edges

## Verified Train Example

`44b6_0113de3b` opened with:

```python
open_dataset(
    TRAIN / "44b6_0113de3b",
    normalize=False,
    require_tracks=True,
    load_image=False,
)
```

Returned:

- scale: `(1.625, 0.40625, 0.40625)`
- image_shape: `(100, 64, 256, 256)`
- image is `None`: `True`
- nodes: `52`
- edges: `50`
- `estimated_number_of_nodes`: `25755`

## Verified Five-Dataset Self-Eval

| dataset | nodes | edges | T_true | edge_tp/fp/fn | division_tp/fp/fn | adj_edge_jaccard |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `44b6_0113de3b` | 52 | 50 | 25755 | 50/0/0 | 0/0/0 | 1.0997980974568045 |
| `44b6_0b24845f` | 51 | 49 | 32795 | 49/0/0 | 0/0/0 | 1.099844488489099 |
| `44b6_0c582fdc` | 71 | 70 | 27958 | 70/0/0 | 0/0/0 | 1.0997460476428929 |
| `44b6_0db75fae` | 157 | 151 | 15335 | 151/0/0 | 0/0/0 | 1.0989761982393218 |
| `44b6_12dfb391` | 788 | 773 | 58672 | 773/0/0 | 1/0/0 | 1.0986569402781565 |

Official `summarise(rows)`:

```json
{
  "edge_jaccard": 1.0,
  "division_jaccard": 1.0,
  "adj_edge_jaccard": 1.0988762387126818,
  "score": 1.198876238712682
}
```

## Sparse-GT Caveat

Annotated GT is sparse, roughly `0.2%` of cells in the verified example (`52` annotated nodes vs `25755` estimated true nodes). Self-eval score above `1.0` is a ceiling artifact of `pred == GT` with a tiny annotated graph; it is not a progress signal.
