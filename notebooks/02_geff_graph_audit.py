"""Audit official GEFF graph stores without loading microscopy image volumes."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.geff import extract_divisions, list_geff_arrays, read_geff_graph  # noqa: E402


TRAIN_ROOT = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")
OUTPUT_PATH = Path("GEFF_GRAPH_AUDIT.md")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in row) + " |")
    return lines


def _sample_array(path: Path, array_path: str, limit: int = 10) -> str:
    try:
        from src.evaluation.geff import _read_array  # noqa: PLC0415

        array = np.asarray(_read_array(path, array_path))
    except (OSError, ValueError, FileNotFoundError, ImportError) as exc:
        return f"unreadable: {exc}"
    if array.size == 0:
        return "[]"
    if array.ndim == 1:
        return np.array2string(array[:limit], threshold=limit)
    return np.array2string(array[:limit], threshold=limit)


def main() -> int:
    lines: list[str] = ["# GEFF Graph Audit", ""]
    lines.append(f"Train root: `{TRAIN_ROOT}`")
    lines.append("")
    if not TRAIN_ROOT.exists():
        lines.append("Train root does not exist in this environment.")
        lines.extend(
            [
                "",
                "Decoder mapping implemented in `src.evaluation.geff`:",
                "",
                "- node ids: `nodes/ids`",
                "- node properties: `nodes/props` named or metadata-labelled fields `t`, `z`, `y`, `x`",
                "- positional properties require metadata such as `property_names`, `columns`, or `properties`",
                "- edge ids: `edges/ids` as source/target pairs",
                "- divisions: inferred from exactly two outgoing child edges",
                "- training `T_true`: GEFF metadata `extra.estimated_number_of_nodes` when available",
                "- adjusted Edge Jaccard: `max(0, edge_jaccard * (1 - 0.1 * (T_pred - T_true) / T_true))`",
            ]
        )
        OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 1

    geff_paths = sorted(TRAIN_ROOT.glob("*.geff"))[:5]
    if not geff_paths:
        lines.append("No `train/*.geff` stores found.")
        OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 1

    for path in geff_paths:
        lines.extend([f"## {path.stem}", ""])
        summaries = list_geff_arrays(path)
        rows = [
            [
                summary.path,
                str(summary.shape),
                str(summary.dtype),
                ", ".join(sorted(summary.attributes)) if summary.attributes else "",
            ]
            for summary in summaries
        ]
        lines.extend(_markdown_table(["array/group", "shape", "dtype", "attribute names"], rows))
        lines.append("")
        for array_path in ("nodes/ids", "nodes/props", "edges/ids", "edges/props"):
            lines.append(f"### {array_path}")
            lines.append("")
            lines.append("```text")
            lines.append(_sample_array(path, array_path))
            lines.append("```")
            lines.append("")
        try:
            graph = read_geff_graph(path)
        except (OSError, ValueError, FileNotFoundError) as exc:
            lines.append(f"Decode error: `{exc}`")
            lines.append("")
            continue
        lines.append("Decoded graph:")
        lines.append("")
        lines.append(f"- nodes: `{len(graph.nodes)}`")
        lines.append(f"- edges: `{len(graph.edges)}`")
        lines.append(f"- inferred divisions: `{len(extract_divisions(graph))}`")
        lines.append(f"- validation errors: `{len(graph.validate())}`")
        if graph.nodes:
            first = graph.nodes[0]
            lines.append(f"- example node: `{first}`")
        if graph.edges:
            lines.append(f"- example edge: `{graph.edges[0]}`")
        lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
