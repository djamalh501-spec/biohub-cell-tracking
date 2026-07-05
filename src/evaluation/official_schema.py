"""Adapters for the official Biohub Kaggle submission and annotation layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schema import CellNode, SequenceGraph, TemporalEdge
from .validator import SubmissionSchema, dataframe_to_graphs, infer_submission_schema


KAGGLE_DATA_ROOT = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
OFFICIAL_SUBMISSION_COLUMNS: tuple[str, ...] = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)
OFFICIAL_GRAPH_SCHEMA = SubmissionSchema(
    node_id="node_id",
    sequence_id="dataset",
    time="t",
    z="z",
    y="y",
    x="x",
    parent_id="source_id",
    child_id="target_id",
    row_type="row_type",
    required_columns=OFFICIAL_SUBMISSION_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class OfficialSampleSubmissionSchema:
    """Observed schema information from the official sample submission."""

    columns: tuple[str, ...]
    row_count: int
    graph_schema: SubmissionSchema
    source_path: Path
    exact_official_columns_verified: bool


@dataclass(frozen=True, slots=True)
class OfficialAnnotationSchema:
    """Observed schema information from training annotation CSV files."""

    path: Path
    columns: tuple[str, ...]
    row_count: int


def is_official_submission_columns(columns: Iterable[object]) -> bool:
    """Return whether columns exactly match the official mixed row format."""

    return tuple(str(column) for column in columns) == OFFICIAL_SUBMISSION_COLUMNS


def read_sample_submission_schema(path: str | Path) -> OfficialSampleSubmissionSchema:
    """Read official ``sample_submission.csv`` and return its exact columns."""

    source_path = Path(path)
    sample = pd.read_csv(source_path)
    columns = tuple(str(column) for column in sample.columns)
    graph_schema = OFFICIAL_GRAPH_SCHEMA if is_official_submission_columns(columns) else infer_submission_schema(sample)
    return OfficialSampleSubmissionSchema(
        columns=columns,
        row_count=len(sample),
        graph_schema=graph_schema,
        source_path=source_path,
        exact_official_columns_verified=is_official_submission_columns(columns),
    )


def regenerate_official_row_ids(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with official global row ids regenerated as ``0..n-1``."""

    output = data.copy()
    output["id"] = range(len(output))
    return output


def official_submission_to_graphs(data: pd.DataFrame) -> dict[str, SequenceGraph]:
    """Convert official mixed node/edge rows into dataset-level graphs."""

    nodes_by_dataset: dict[str, list[CellNode]] = {}
    edges_by_dataset: dict[str, list[TemporalEdge]] = {}
    for _, row in data.iterrows():
        dataset = str(row["dataset"])
        row_type = str(row["row_type"]).strip().lower()
        nodes_by_dataset.setdefault(dataset, [])
        edges_by_dataset.setdefault(dataset, [])
        if row_type == "node":
            nodes_by_dataset[dataset].append(
                CellNode(
                    node_id=str(int(row["node_id"])),
                    sequence_id=dataset,
                    time=int(row["t"]),
                    z=float(row["z"]),
                    y=float(row["y"]),
                    x=float(row["x"]),
                )
            )
        elif row_type == "edge":
            edges_by_dataset[dataset].append(
                TemporalEdge(parent_id=str(int(row["source_id"])), child_id=str(int(row["target_id"])))
            )
    return {
        dataset: SequenceGraph(
            sequence_id=dataset,
            nodes=tuple(nodes_by_dataset.get(dataset, ())),
            edges=tuple(edges_by_dataset.get(dataset, ())),
        )
        for dataset in sorted(set(nodes_by_dataset) | set(edges_by_dataset))
    }


def read_submission_csv(
    path: str | Path,
    sample_submission_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, SequenceGraph], OfficialSampleSubmissionSchema, tuple[str, ...]]:
    """Read a submission CSV and convert official rows into sequence graphs."""

    official_schema = read_sample_submission_schema(sample_submission_path)
    submission = pd.read_csv(path)
    if official_schema.exact_official_columns_verified:
        graphs = official_submission_to_graphs(submission)
        warnings: tuple[str, ...] = ()
    else:
        graphs, warning_list = dataframe_to_graphs(submission, official_schema.graph_schema)
        warnings = tuple(warning_list)
    return submission, graphs, official_schema, tuple(warnings)


def find_training_annotation_files(data_root: str | Path = KAGGLE_DATA_ROOT) -> tuple[Path, ...]:
    """Return likely official training annotation stores without reading image volumes."""

    root = Path(data_root)
    if not root.exists():
        return ()
    candidates: list[Path] = []
    for path in root.rglob("*"):
        lowered = path.name.lower()
        parts = {part.lower() for part in path.parts}
        if path.suffix.lower() == ".geff":
            candidates.append(path)
        elif path.suffix.lower() == ".csv" and (
            "train" in parts or "annotation" in lowered or "label" in lowered or "track" in lowered
        ):
            if path.name.lower() != "sample_submission.csv":
                candidates.append(path)
    return tuple(sorted(candidates))


def read_training_annotations(
    data_root: str | Path = KAGGLE_DATA_ROOT,
) -> tuple[OfficialAnnotationSchema, ...]:
    """Read training annotation CSV headers and GEFF store summaries.

    GEFF stores are directory-backed graph stores with arrays such as ``nodes/ids``,
    ``nodes/props``, ``edges/ids``, and ``edges/props``. This function records store
    paths without loading full microscopy volumes.
    """

    schemas: list[OfficialAnnotationSchema] = []
    for path in find_training_annotation_files(data_root):
        if path.suffix.lower() == ".geff":
            schemas.append(OfficialAnnotationSchema(path=path, columns=("nodes/ids", "nodes/props", "edges/ids", "edges/props"), row_count=-1))
            continue
        frame = pd.read_csv(path)
        schemas.append(
            OfficialAnnotationSchema(
                path=path,
                columns=tuple(str(column) for column in frame.columns),
                row_count=len(frame),
            )
        )
    return tuple(schemas)
