"""Bridge between official submission rows and graph objects.

The bridge uses the lightweight local ``SequenceGraph`` representation as the fallback
graph type. When the official ``tracksdata`` package is installed, callers may convert
from these validated graphs into tracksdata-specific objects outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from .schema import CellNode, SequenceGraph, TemporalEdge
from .validator import (
    OFFICIAL_SUBMISSION_COLUMNS,
    _validate_official_rows,
    has_official_columns,
    official_dataframe_to_graphs,
    regenerate_official_row_ids,
)


try:  # pragma: no cover - depends on optional official package.
    import tracksdata as _tracksdata  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - local default.
    _tracksdata = None


@dataclass(frozen=True, slots=True)
class BridgeResult:
    graphs: dict[str, SequenceGraph]
    tracksdata_available: bool


def _validate_official_submission_df(df: pd.DataFrame) -> None:
    if not has_official_columns(df):
        raise ValueError(f"submission columns must be exactly {OFFICIAL_SUBMISSION_COLUMNS}, got {tuple(df.columns)}")
    expected_ids = list(range(len(df)))
    actual_ids = [int(value) for value in df["id"]]
    if actual_ids != expected_ids:
        raise ValueError("submission id column must be sequential row ids 0..len(submission)-1")
    row_errors = _validate_official_rows(df)
    if row_errors:
        raise ValueError("official submission row validation failed: " + "; ".join(row_errors))


def _build_official_tracksdata_graphs(df: pd.DataFrame):
    if _tracksdata is None:
        return None
    try:
        import polars as pl  # type: ignore[import-not-found]
    except ImportError:
        return None

    graphs = {}
    for dataset, dataset_df in df.groupby("dataset", sort=True):
        graph = _tracksdata.graph.InMemoryGraph()
        graph.add_node_attr_key("t", pl.Int64, default_value=0)
        for key in ("z", "y", "x"):
            graph.add_node_attr_key(key, pl.Float64, default_value=0.0)
        node_rows = dataset_df[dataset_df["row_type"] == "node"].sort_values(["t", "node_id"])
        nodes = [
            {
                "t": int(row.t),
                "z": float(row.z),
                "y": float(row.y),
                "x": float(row.x),
            }
            for row in node_rows.itertuples(index=False)
        ]
        indices = [int(row.node_id) for row in node_rows.itertuples(index=False)]
        graph.bulk_add_nodes(nodes, indices=indices)
        edge_rows = dataset_df[dataset_df["row_type"] == "edge"].sort_values(["source_id", "target_id"])
        edges = [
            {"source_id": int(row.source_id), "target_id": int(row.target_id)}
            for row in edge_rows.itertuples(index=False)
        ]
        graph.bulk_add_edges(edges)
        graphs[str(dataset)] = graph
    return graphs


def submission_df_to_tracksdata(df: pd.DataFrame):
    """Convert official submission rows into dataset-scoped graph objects.

    The returned objects are local ``SequenceGraph`` instances when the optional
    ``tracksdata`` package is unavailable. The function name intentionally mirrors the
    official bridge goal while preserving a dependency-light local path.
    """

    _validate_official_submission_df(df)
    official_graphs = _build_official_tracksdata_graphs(df)
    if official_graphs is not None:
        return official_graphs

    graphs, warnings = official_dataframe_to_graphs(df)
    errors = [error for graph in graphs.values() for error in graph.validate()]
    if warnings:
        raise ValueError("official submission parse warnings: " + "; ".join(warnings))
    if errors:
        raise ValueError("official submission graph validation failed: " + "; ".join(errors))
    return graphs


def _polars_rows(frame: Any) -> list[dict[str, Any]]:
    if hasattr(frame, "iter_rows"):
        return list(frame.iter_rows(named=True))
    if hasattr(frame, "to_dict"):
        data = frame.to_dict(orient="records")
        return list(data)
    raise TypeError(f"unsupported graph attribute table type: {type(frame)!r}")


def _column(row: Mapping[str, Any], candidates: tuple[str, ...]) -> Any:
    for candidate in candidates:
        if candidate in row:
            return row[candidate]
    raise KeyError(f"none of the candidate columns {candidates} were found in row keys {tuple(row)}")


def _tracksdata_rows(dataset: str, graph: Any) -> list[dict[str, object]]:
    node_rows = _polars_rows(graph.node_attrs())
    edge_rows = _polars_rows(graph.edge_attrs())
    rows: list[dict[str, object]] = []
    for row in node_rows:
        rows.append(
            {
                "id": -1,
                "dataset": dataset,
                "row_type": "node",
                "node_id": int(_column(row, ("node_id", "id"))),
                "t": int(_column(row, ("t", "time"))),
                "z": int(round(float(_column(row, ("z",))))),
                "y": int(round(float(_column(row, ("y",))))),
                "x": int(round(float(_column(row, ("x",))))),
                "source_id": -1,
                "target_id": -1,
            }
        )
    for row in edge_rows:
        rows.append(
            {
                "id": -1,
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(_column(row, ("source_id", "source", "edge_source"))),
                "target_id": int(_column(row, ("target_id", "target", "edge_target"))),
            }
        )
    return rows


def tracksdata_to_submission_df(graphs: Mapping[str, Any]) -> pd.DataFrame:
    """Convert dataset-scoped graphs into the exact official submission CSV schema."""

    rows: list[dict[str, object]] = []
    for dataset, graph in sorted(graphs.items()):
        if isinstance(graph, SequenceGraph) and graph.sequence_id != dataset:
            raise ValueError(f"graph key {dataset!r} does not match graph.sequence_id {graph.sequence_id!r}")
        if isinstance(graph, SequenceGraph):
            for node in sorted(graph.nodes, key=lambda item: (item.time, int(item.node_id))):
                rows.append(
                    {
                        "id": -1,
                        "dataset": dataset,
                        "row_type": "node",
                        "node_id": int(node.node_id),
                        "t": int(node.time),
                        "z": int(round(node.z)),
                        "y": int(round(node.y)),
                        "x": int(round(node.x)),
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
            for edge in sorted(graph.edges, key=lambda item: (int(item.parent_id), int(item.child_id))):
                rows.append(
                    {
                        "id": -1,
                        "dataset": dataset,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": int(edge.parent_id),
                        "target_id": int(edge.child_id),
                    }
                )
        elif hasattr(graph, "node_attrs") and hasattr(graph, "edge_attrs"):
            rows.extend(_tracksdata_rows(dataset, graph))
        else:
            raise TypeError(f"unsupported graph type for dataset {dataset!r}: {type(graph)!r}")
    df = pd.DataFrame(rows, columns=OFFICIAL_SUBMISSION_COLUMNS)
    return regenerate_official_row_ids(df)


def sequence_graph_from_records(
    dataset: str,
    nodes: list[tuple[int, int, float, float, float]],
    edges: list[tuple[int, int]],
) -> SequenceGraph:
    """Small helper for tests and examples using official dataset-local IDs."""

    return SequenceGraph(
        sequence_id=dataset,
        nodes=tuple(
            CellNode(str(node_id), dataset, int(time), float(z), float(y), float(x))
            for node_id, time, z, y, x in nodes
        ),
        edges=tuple(TemporalEdge(str(source_id), str(target_id)) for source_id, target_id in edges),
    )
