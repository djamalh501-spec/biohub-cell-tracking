"""Bridge between official submission rows and graph objects.

The bridge uses the lightweight local ``SequenceGraph`` representation as the fallback
graph type. When the official ``tracksdata`` package is installed, callers may convert
from these validated graphs into tracksdata-specific objects outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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


def submission_df_to_tracksdata(df: pd.DataFrame) -> dict[str, SequenceGraph]:
    """Convert official submission rows into dataset-scoped graph objects.

    The returned objects are local ``SequenceGraph`` instances when the optional
    ``tracksdata`` package is unavailable. The function name intentionally mirrors the
    official bridge goal while preserving a dependency-light local path.
    """

    _validate_official_submission_df(df)
    graphs, warnings = official_dataframe_to_graphs(df)
    errors = [error for graph in graphs.values() for error in graph.validate()]
    if warnings:
        raise ValueError("official submission parse warnings: " + "; ".join(warnings))
    if errors:
        raise ValueError("official submission graph validation failed: " + "; ".join(errors))
    return graphs


def tracksdata_to_submission_df(graphs: Mapping[str, SequenceGraph]) -> pd.DataFrame:
    """Convert dataset-scoped graphs into the exact official submission CSV schema."""

    rows: list[dict[str, object]] = []
    for dataset, graph in sorted(graphs.items()):
        if graph.sequence_id != dataset:
            raise ValueError(f"graph key {dataset!r} does not match graph.sequence_id {graph.sequence_id!r}")
        for node in graph.nodes:
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
        for edge in graph.edges:
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
