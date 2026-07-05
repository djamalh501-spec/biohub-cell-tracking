"""Submission CSV schema inference and validation CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

from .schema import CellNode, SequenceGraph, TemporalEdge

LOGGER = logging.getLogger(__name__)


NODE_ID_CANDIDATES = ("node_id", "cell_id", "track_id", "id", "object_id")
SEQUENCE_CANDIDATES = ("dataset", "sequence_id", "embryo_id", "video_id", "series_id", "sample_id", "experiment_id")
TIME_CANDIDATES = ("time", "t", "frame", "timepoint")
Z_CANDIDATES = ("z", "z_voxel", "z_px", "center_z", "centroid_z")
Y_CANDIDATES = ("y", "y_voxel", "y_px", "center_y", "centroid_y")
X_CANDIDATES = ("x", "x_voxel", "x_px", "center_x", "centroid_x")
PARENT_CANDIDATES = ("parent_id", "parent", "predecessor_id", "source_id", "from_id")
CHILD_CANDIDATES = ("child_id", "child", "successor_id", "target_id", "to_id")
ROW_TYPE_CANDIDATES = ("type", "row_type", "record_type", "kind")
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


@dataclass(frozen=True, slots=True)
class SubmissionSchema:
    node_id: str
    sequence_id: str | None
    time: str
    z: str
    y: str
    x: str
    parent_id: str | None
    child_id: str | None
    row_type: str | None
    required_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    schema: SubmissionSchema
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    sample_row_count: int | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def _normalise(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _find_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    normalised = {_normalise(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]
    for column in columns:
        lowered = _normalise(column)
        if any(lowered.endswith(f"_{candidate}") for candidate in candidates):
            return column
    return None


def infer_submission_schema(sample_submission: pd.DataFrame) -> SubmissionSchema:
    """Infer a submission schema from a sample submission header."""

    columns = tuple(str(column) for column in sample_submission.columns)
    node_id = _find_column(columns, NODE_ID_CANDIDATES)
    sequence_id = _find_column(columns, SEQUENCE_CANDIDATES)
    time = _find_column(columns, TIME_CANDIDATES)
    z = _find_column(columns, Z_CANDIDATES)
    y = _find_column(columns, Y_CANDIDATES)
    x = _find_column(columns, X_CANDIDATES)
    parent_id = _find_column(columns, PARENT_CANDIDATES)
    child_id = _find_column(columns, CHILD_CANDIDATES)
    row_type = _find_column(columns, ROW_TYPE_CANDIDATES)

    missing = [
        label
        for label, value in (
            ("node id", node_id),
            ("time", time),
            ("z coordinate", z),
            ("y coordinate", y),
            ("x coordinate", x),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "could not infer required sample submission columns for: "
            + ", ".join(missing)
            + f". Available columns: {list(columns)}"
        )

    return SubmissionSchema(
        node_id=node_id,
        sequence_id=sequence_id,
        time=time,
        z=z,
        y=y,
        x=x,
        parent_id=parent_id,
        child_id=child_id,
        row_type=row_type,
        required_columns=columns,
    )


def official_submission_schema() -> SubmissionSchema:
    """Return the official mixed node/edge submission schema."""

    return SubmissionSchema(
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


def has_official_columns(frame: pd.DataFrame) -> bool:
    return tuple(str(column) for column in frame.columns) == OFFICIAL_SUBMISSION_COLUMNS


def official_dataset_ids_from_sample(sample: pd.DataFrame) -> set[str]:
    """Derive allowed dataset IDs from an official sample submission."""

    if not has_official_columns(sample):
        return set()
    return {_value_to_id(value) for value in sample["dataset"].dropna()}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _value_to_id(value: object) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _row_is_node(row: pd.Series, schema: SubmissionSchema) -> bool:
    if schema.row_type is not None and not _is_missing(row.get(schema.row_type)):
        row_type = str(row[schema.row_type]).strip().lower()
        if row_type in {"edge", "edges", "link", "links"}:
            return False
    return not _is_missing(row.get(schema.node_id))


def _row_is_edge(row: pd.Series, schema: SubmissionSchema) -> bool:
    if schema.parent_id is None:
        return False
    if schema.child_id is not None:
        return not _is_missing(row.get(schema.parent_id)) and not _is_missing(row.get(schema.child_id))
    return not _is_missing(row.get(schema.parent_id)) and not _is_missing(row.get(schema.node_id))


def dataframe_to_graphs(
    data: pd.DataFrame,
    schema: SubmissionSchema,
) -> tuple[dict[str, SequenceGraph], list[str]]:
    """Parse submission rows into sequence graphs and return warnings."""

    warnings: list[str] = []
    nodes_by_sequence: dict[str, list[CellNode]] = {}
    edges_by_sequence: dict[str, list[TemporalEdge]] = {}

    for index, row in data.iterrows():
        sequence_id = _value_to_id(row[schema.sequence_id]) if schema.sequence_id else "default"
        nodes_by_sequence.setdefault(sequence_id, [])
        edges_by_sequence.setdefault(sequence_id, [])
        if _row_is_node(row, schema):
            try:
                node = CellNode(
                    node_id=_value_to_id(row[schema.node_id]),
                    sequence_id=sequence_id,
                    time=int(row[schema.time]),
                    z=float(row[schema.z]),
                    y=float(row[schema.y]),
                    x=float(row[schema.x]),
                )
            except (TypeError, ValueError) as exc:
                warnings.append(f"row {index}: could not parse node fields: {exc}")
            else:
                nodes_by_sequence[sequence_id].append(node)
        if _row_is_edge(row, schema):
            child_value = row[schema.child_id] if schema.child_id is not None else row[schema.node_id]
            edges_by_sequence[sequence_id].append(
                TemporalEdge(parent_id=_value_to_id(row[schema.parent_id]), child_id=_value_to_id(child_value))
            )

    return (
        {
            sequence_id: SequenceGraph(
                sequence_id=sequence_id,
                nodes=tuple(nodes_by_sequence.get(sequence_id, ())),
                edges=tuple(edges_by_sequence.get(sequence_id, ())),
            )
            for sequence_id in sorted(set(nodes_by_sequence) | set(edges_by_sequence))
        },
        warnings,
    )


def _as_int(value: object) -> int:
    if _is_missing(value):
        raise ValueError("missing integer")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"not an integer: {value!r}")
    return int(number)


def _as_finite_float(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"not finite: {value!r}")
    return number


def official_dataframe_to_graphs(data: pd.DataFrame) -> tuple[dict[str, SequenceGraph], list[str]]:
    """Parse official mixed node/edge rows into dataset-scoped graphs."""

    warnings: list[str] = []
    nodes_by_dataset: dict[str, list[CellNode]] = {}
    edges_by_dataset: dict[str, list[TemporalEdge]] = {}
    for index, row in data.iterrows():
        dataset = _value_to_id(row["dataset"])
        row_type = str(row["row_type"]).strip().lower()
        nodes_by_dataset.setdefault(dataset, [])
        edges_by_dataset.setdefault(dataset, [])
        try:
            if row_type == "node":
                nodes_by_dataset[dataset].append(
                    CellNode(
                        node_id=str(_as_int(row["node_id"])),
                        sequence_id=dataset,
                        time=_as_int(row["t"]),
                        z=_as_finite_float(row["z"]),
                        y=_as_finite_float(row["y"]),
                        x=_as_finite_float(row["x"]),
                    )
                )
            elif row_type == "edge":
                edges_by_dataset[dataset].append(
                    TemporalEdge(
                        parent_id=str(_as_int(row["source_id"])),
                        child_id=str(_as_int(row["target_id"])),
                    )
                )
            else:
                warnings.append(f"row {index}: unknown row_type {row['row_type']!r}")
        except (TypeError, ValueError) as exc:
            warnings.append(f"row {index}: could not parse official row: {exc}")

    return (
        {
            dataset: SequenceGraph(
                sequence_id=dataset,
                nodes=tuple(nodes_by_dataset.get(dataset, ())),
                edges=tuple(edges_by_dataset.get(dataset, ())),
            )
            for dataset in sorted(set(nodes_by_dataset) | set(edges_by_dataset))
        },
        warnings,
    )


def regenerate_official_row_ids(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with official ``id`` values reset to ``0..len(data)-1``."""

    output = data.copy()
    output["id"] = range(len(output))
    return output


def _validate_official_rows(data: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    valid_row_types = {"node", "edge"}
    for index, row in data.iterrows():
        row_type = str(row["row_type"]).strip().lower()
        dataset = _value_to_id(row["dataset"])
        if row_type not in valid_row_types:
            errors.append(f"row {index}: row_type must be 'node' or 'edge', got {row['row_type']!r}")
            continue
        try:
            node_id = _as_int(row["node_id"])
            time = _as_int(row["t"])
            z = _as_finite_float(row["z"])
            y = _as_finite_float(row["y"])
            x = _as_finite_float(row["x"])
            source_id = _as_int(row["source_id"])
            target_id = _as_int(row["target_id"])
        except (TypeError, ValueError) as exc:
            errors.append(f"row {index}: official numeric fields are invalid: {exc}")
            continue

        if row_type == "node":
            if node_id == -1:
                errors.append(f"row {index}: node row node_id must not be -1")
            if time == -1 or z == -1 or y == -1 or x == -1:
                errors.append(f"row {index}: node row t, z, y, x must be finite values other than -1")
            if source_id != -1:
                errors.append(f"row {index}: node row source_id must be -1")
            if target_id != -1:
                errors.append(f"row {index}: node row target_id must be -1")
        else:
            if node_id != -1:
                errors.append(f"row {index}: edge row node_id must be -1")
            if time != -1 or z != -1 or y != -1 or x != -1:
                errors.append(f"row {index}: edge row t, z, y, x must all be -1")
            if source_id == -1:
                errors.append(f"row {index}: edge row source_id must not be -1")
            if target_id == -1:
                errors.append(f"row {index}: edge row target_id must not be -1")
            if source_id == target_id and source_id != -1:
                errors.append(f"row {index}: edge source_id and target_id must differ within dataset {dataset!r}")
    return errors


def validate_submission(
    submission_path: str | Path,
    sample_submission_path: str | Path,
    *,
    strict: bool = False,
    valid_sequence_ids: set[str] | None = None,
) -> ValidationResult:
    """Validate a submission against the official sample-submission schema.

    The official sample file provides the exact required columns. Column-to-graph-role
    inference is retained only as a fallback mapping layer because the official CSV is
    not available in this repository copy.
    """

    sample = pd.read_csv(sample_submission_path)
    submission = pd.read_csv(submission_path)
    official_mode = has_official_columns(sample)
    schema = official_submission_schema() if official_mode else infer_submission_schema(sample)
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [column for column in schema.required_columns if column not in submission.columns]
    if missing_columns:
        errors.append(f"submission is missing required columns: {missing_columns}")
        return ValidationResult(
            schema=schema,
            errors=tuple(errors),
            warnings=tuple(warnings),
            sample_row_count=len(sample),
        )

    extra_columns = [column for column in submission.columns if column not in schema.required_columns]
    if extra_columns:
        warnings.append(f"submission has extra columns not present in sample submission: {extra_columns}")

    if tuple(str(column) for column in submission.columns[: len(schema.required_columns)]) != schema.required_columns:
        warnings.append("submission column order differs from sample_submission.csv")

    if len(sample) > 0 and len(submission) == 0:
        errors.append("submission has zero rows while sample_submission.csv is non-empty")

    if official_mode:
        expected_ids = list(range(len(submission)))
        actual_ids: list[int] = []
        for value in submission["id"]:
            try:
                actual_ids.append(_as_int(value))
            except (TypeError, ValueError):
                actual_ids.append(-999999)
        if actual_ids != expected_ids:
            errors.append("official id column must be regenerated as sequential row ids 0..len(submission)-1")
        errors.extend(_validate_official_rows(submission))
        graphs, parse_warnings = official_dataframe_to_graphs(submission)
        if valid_sequence_ids is None:
            valid_sequence_ids = official_dataset_ids_from_sample(sample)
    else:
        node_mask = submission.apply(lambda row: _row_is_node(row, schema), axis=1)
        node_rows = submission.loc[node_mask]
        required_non_null = [schema.node_id, schema.time, schema.z, schema.y, schema.x]
        if schema.sequence_id is not None:
            required_non_null.append(schema.sequence_id)
        for column in required_non_null:
            if node_rows[column].isna().any():
                errors.append(f"node column {column!r} contains null values")

        for column in (schema.time, schema.z, schema.y, schema.x):
            numeric = pd.to_numeric(node_rows[column], errors="coerce")
            if numeric.isna().any():
                errors.append(f"node column {column!r} contains non-numeric values")
            if not numeric.map(lambda value: math.isfinite(float(value)) if not pd.isna(value) else False).all():
                errors.append(f"node column {column!r} contains non-finite values")

        graphs, parse_warnings = dataframe_to_graphs(submission, schema)
    warnings.extend(parse_warnings)
    for graph in graphs.values():
        errors.extend(graph.validate())
        if valid_sequence_ids is not None and graph.sequence_id not in valid_sequence_ids:
            errors.append(f"unknown sequence/sample id {graph.sequence_id!r}")

    if official_mode:
        warnings.append("using official Biohub mixed node/edge submission schema")
    elif schema.parent_id is None:
        warnings.append("no parent/source column was inferred; edge validation was limited to node rows")
    elif schema.child_id is None:
        warnings.append(
            f"using parent column {schema.parent_id!r} and node column {schema.node_id!r} as parent->child edges"
        )

    if strict and warnings:
        errors.extend(f"strict warning: {warning}" for warning in warnings)

    return ValidationResult(
        schema=schema,
        errors=tuple(errors),
        warnings=tuple(warnings),
        sample_row_count=len(sample),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a Biohub lineage submission CSV.")
    parser.add_argument("--submission", required=True, type=Path, help="Path to submission.csv")
    parser.add_argument("--sample-submission", required=True, type=Path, help="Path to sample_submission.csv")
    parser.add_argument("--test-metadata", type=Path, help="Optional test metadata CSV with valid sequence IDs")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as validation failures")
    return parser


def _read_valid_sequence_ids(metadata_path: Path | None) -> set[str] | None:
    if metadata_path is None:
        return None
    metadata = pd.read_csv(metadata_path)
    sequence_column = _find_column(tuple(str(column) for column in metadata.columns), SEQUENCE_CANDIDATES)
    if sequence_column is None:
        raise ValueError(f"could not infer sequence/sample id column from metadata: {list(metadata.columns)}")
    return {_value_to_id(value) for value in metadata[sequence_column].dropna()}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        valid_sequence_ids = _read_valid_sequence_ids(args.test_metadata)
        result = validate_submission(
            args.submission,
            args.sample_submission,
            strict=args.strict,
            valid_sequence_ids=valid_sequence_ids,
        )
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2

    LOGGER.info("inferred schema: %s", result.schema)
    for warning in result.warnings:
        LOGGER.warning("%s", warning)
    for error in result.errors:
        LOGGER.error("%s", error)
    if not result.ok:
        LOGGER.error("submission validation failed with %d error(s)", len(result.errors))
        return 1
    LOGGER.info("submission validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
