# Official Schema Audit

This file records the Kaggle Notebook official audit update supplied by the user.

## sample_submission.csv

Shape:

```text
(20, 10)
```

Columns:

```text
id, dataset, row_type, node_id, t, z, y, x, source_id, target_id
```

## Row Format

Node row example:

```csv
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6_0113de3b,node,1,0,32,128,128,-1,-1
```

Edge row example:

```csv
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
3,44b6_0113de3b,edge,-1,-1,-1,-1,-1,1,2
```

## Test Datasets

- `44b6_0113de3b`
- `44b6_0b24845f`
- `6bba_05b6850b`
- `6bba_05db0fb1`

## Train Stores

- `train/<dataset>.zarr`
- `train/<dataset>.geff`

GEFF paths include:

- `nodes/ids`
- `nodes/props`
- `edges/ids`
- `edges/props`
