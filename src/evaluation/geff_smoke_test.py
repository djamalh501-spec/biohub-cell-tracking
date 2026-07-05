"""Optional integration smoke tests for official GEFF training annotations."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

from .geff import extract_divisions, read_geff_graph
from .matching import match_nodes
from .metrics import division_jaccard, raw_edge_jaccard


def _coordinate_ranges(graph) -> str:
    if not graph.nodes:
        return "empty graph"
    times = [node.time for node in graph.nodes]
    zs = [node.z for node in graph.nodes]
    ys = [node.y for node in graph.nodes]
    xs = [node.x for node in graph.nodes]
    return (
        f"t=[{min(times)}, {max(times)}], "
        f"z=[{min(zs)}, {max(zs)}], "
        f"y=[{min(ys)}, {max(ys)}], "
        f"x=[{min(xs)}, {max(xs)}]"
    )


def _edge_skip_count(graph) -> int:
    nodes = graph.node_by_id
    return sum(1 for edge in graph.edges if nodes[edge.child_id].time - nodes[edge.parent_id].time > 1)


def _multiple_parent_children(graph) -> list[str]:
    parent_counts = Counter(edge.child_id for edge in graph.edges)
    return sorted(child_id for child_id, count in parent_counts.items() if count > 1)


def _run_self_evaluation(graph) -> str:
    matches = match_nodes(graph.nodes, graph.nodes)
    edge_result = raw_edge_jaccard(graph, graph, matches)
    division_result = division_jaccard(graph, graph, matches)
    precision = 1.0 if not graph.nodes else len(matches.gt_to_pred) / len(graph.nodes)
    recall = 1.0 if not graph.nodes else len(matches.gt_to_pred) / len(graph.nodes)
    return (
        f"self_eval: node_precision={precision:.6f}, node_recall={recall:.6f}, "
        f"raw_edge_jaccard={edge_result.raw_score:.6f}, "
        f"division_jaccard_assumed={division_result.score:.6f}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test official Biohub GEFF graphs.")
    parser.add_argument(
        "--train-root",
        required=True,
        type=Path,
        help="Path to /kaggle/input/competitions/biohub-cell-tracking-during-development/train",
    )
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of GEFF stores to inspect")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    geff_paths = sorted(args.train_root.glob("*.geff"))[: args.limit]
    if not geff_paths:
        print(f"no GEFF stores found under {args.train_root}")
        return 1

    failures = 0
    for path in geff_paths:
        print(f"dataset={path.stem}")
        try:
            graph = read_geff_graph(path)
        except (OSError, ValueError, FileNotFoundError) as exc:
            failures += 1
            print(f"  ERROR: {exc}")
            continue
        divisions = extract_divisions(graph)
        validation_errors = graph.validate()
        multiple_parent_children = _multiple_parent_children(graph)
        print(f"  nodes={len(graph.nodes)}")
        print(f"  edges={len(graph.edges)}")
        print(f"  divisions={len(divisions)}")
        print(f"  ranges={_coordinate_ranges(graph)}")
        print(f"  validation_errors={len(validation_errors)}")
        for error in validation_errors[:10]:
            print(f"    {error}")
        print(f"  edges_skip_frames={_edge_skip_count(graph)}")
        print(f"  children_with_multiple_parents={len(multiple_parent_children)}")
        if multiple_parent_children:
            print(f"    examples={multiple_parent_children[:10]}")
        print(f"  {_run_self_evaluation(graph)}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
