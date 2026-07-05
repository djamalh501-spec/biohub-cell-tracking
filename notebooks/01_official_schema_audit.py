"""Audit official Kaggle schema without loading full microscopy volumes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_ROOT = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
OUTPUT_PATH = Path("OFFICIAL_SCHEMA_AUDIT.md")


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in row) + " |")
    return lines


def safe_csv_summary(path: Path) -> tuple[list[str], int, pd.DataFrame]:
    frame = pd.read_csv(path)
    return [str(column) for column in frame.columns], len(frame), frame.head(20)


def inspect_zarr_metadata(root: Path) -> list[tuple[str, str]]:
    summaries: list[tuple[str, str]] = []
    for zarr_dir in sorted(root.rglob("*.zarr")):
        details: list[str] = []
        for metadata_name in (".zarray", ".zattrs"):
            for metadata_path in sorted(zarr_dir.rglob(metadata_name)):
                relative = metadata_path.relative_to(zarr_dir)
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    details.append(f"{relative}: unreadable ({exc})")
                    continue
                shape = metadata.get("shape")
                chunks = metadata.get("chunks")
                dtype = metadata.get("dtype")
                if shape is not None or chunks is not None or dtype is not None:
                    details.append(f"{relative}: shape={shape}, chunks={chunks}, dtype={dtype}")
                else:
                    details.append(f"{relative}: keys={sorted(metadata)}")
        summaries.append((str(zarr_dir), "<br>".join(details) if details else "no .zarray/.zattrs found"))
    return summaries


def inspect_geff_layout(root: Path) -> list[tuple[str, str]]:
    summaries: list[tuple[str, str]] = []
    for geff_dir in sorted(path for path in root.rglob("*.geff") if path.is_dir()):
        expected = ["nodes/ids", "nodes/props", "edges/ids", "edges/props"]
        details = []
        for relative in expected:
            path = geff_dir / relative
            details.append(f"{relative}: {'present' if path.exists() else 'missing'}")
        summaries.append((str(geff_dir), "<br>".join(details)))
    return summaries


def main() -> int:
    lines: list[str] = ["# Official Schema Audit", ""]
    lines.append(f"Data root: `{DATA_ROOT}`")
    lines.append("")
    if not DATA_ROOT.exists():
        lines.append("Data root does not exist in this environment.")
        OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 1

    top_level = sorted(DATA_ROOT.iterdir(), key=lambda path: path.name.lower())
    lines.extend(["## Top-level directory structure", ""])
    for path in top_level:
        suffix = "/" if path.is_dir() else ""
        lines.append(f"- `{path.name}{suffix}`")
    lines.append("")

    csv_paths = sorted(DATA_ROOT.rglob("*.csv"))
    lines.extend(["## CSV files", ""])
    rows: list[list[str]] = []
    for path in csv_paths:
        columns, row_count, _ = safe_csv_summary(path)
        rows.append([str(path.relative_to(DATA_ROOT)), str(row_count), ", ".join(columns)])
    lines.extend(markdown_table(["path", "rows", "columns"], rows) if rows else ["No CSV files found."])
    lines.append("")

    sample_path = DATA_ROOT / "sample_submission.csv"
    lines.extend(["## sample_submission.csv", ""])
    if sample_path.exists():
        columns, row_count, head = safe_csv_summary(sample_path)
        lines.append(f"Rows: `{row_count}`")
        lines.append(f"Columns: `{columns}`")
        lines.append("")
        lines.append("First 20 rows:")
        lines.append("")
        lines.append("```text")
        lines.append(head.to_string(index=False))
        lines.append("```")
    else:
        lines.append("Not found.")
    lines.append("")

    annotation_rows: list[list[str]] = []
    metadata_rows: list[list[str]] = []
    documentation_rows: list[list[str]] = []
    for path in sorted(DATA_ROOT.rglob("*")):
        lowered = path.name.lower()
        relative = str(path.relative_to(DATA_ROOT))
        if path.is_dir():
            if path.suffix.lower() == ".geff":
                annotation_rows.append([relative, "GEFF store", "nodes/ids, nodes/props, edges/ids, edges/props"])
            continue
        if lowered.endswith(".csv") and any(token in lowered for token in ("annotation", "label", "track", "train")):
            columns, row_count, _ = safe_csv_summary(path)
            annotation_rows.append([relative, str(row_count), ", ".join(columns)])
        if any(token in lowered for token in ("metadata", "meta")):
            metadata_rows.append([relative, str(path.stat().st_size)])
        if lowered.endswith((".md", ".txt", ".json", ".yaml", ".yml")) and any(
            token in lowered for token in ("metric", "readme", "description", "doc", "evaluation")
        ):
            documentation_rows.append([relative, str(path.stat().st_size)])

    lines.extend(["## Training annotation files", ""])
    lines.extend(markdown_table(["path", "rows", "columns"], annotation_rows) if annotation_rows else ["None detected."])
    lines.append("")
    lines.extend(["## Metadata files", ""])
    lines.extend(markdown_table(["path", "bytes"], metadata_rows) if metadata_rows else ["None detected."])
    lines.append("")
    lines.extend(["## Documentation or metric-related files", ""])
    lines.extend(markdown_table(["path", "bytes"], documentation_rows) if documentation_rows else ["None detected."])
    lines.append("")

    zarr_rows = [[path, details] for path, details in inspect_zarr_metadata(DATA_ROOT)]
    lines.extend(["## Zarr layouts", ""])
    lines.extend(markdown_table(["path", "metadata"], zarr_rows) if zarr_rows else ["No Zarr directories found."])
    lines.append("")

    geff_rows = [[path, details] for path, details in inspect_geff_layout(DATA_ROOT)]
    lines.extend(["## GEFF layouts", ""])
    lines.extend(markdown_table(["path", "layout"], geff_rows) if geff_rows else ["No GEFF directories found."])
    lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
