# GEFF Graph Audit

Train root: `\kaggle\input\competitions\biohub-cell-tracking-during-development\train`

Train root does not exist in this environment.

Decoder mapping implemented in `src.evaluation.geff`:

- node ids: `nodes/ids`
- node properties: `nodes/props` named or metadata-labelled fields `t`, `z`, `y`, `x`
- positional properties require metadata such as `property_names`, `columns`, or `properties`
- edge ids: `edges/ids` as source/target pairs
- divisions: inferred from exactly two outgoing child edges
- training `T_true`: read from GEFF metadata `extra.estimated_number_of_nodes` when available
- adjusted Edge Jaccard: `max(0, edge_jaccard * (1 - 0.1 * (T_pred - T_true) / T_true))`

Local synthetic fixture tests verify:

- positional and structured `nodes/props`
- `edges/ids` source/target mapping
- missing edge references are rejected
- submission round-trip regenerates row ids
- graph self-evaluation gives perfect raw edge and division scores when `pred == gt`
