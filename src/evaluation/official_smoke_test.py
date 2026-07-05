"""Optional official-style smoke test for real Kaggle training GEFF datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .geff import extract_divisions, read_geff_graph
from .official_metric import evaluate_one_dataset, read_estimated_number_of_nodes


try:  # pragma: no cover - optional official package.
    from tracking_cellmot.io import open_dataset  # type: ignore[import-not-found]
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - expected locally.
    open_dataset = None
    evaluate = None
    node_recall = None
    per_sample_metrics = None
    summarise = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official-style smoke test on Biohub train GEFF stores.")
    parser.add_argument(
        "--train-root",
        required=True,
        type=Path,
        help="Path to /kaggle/input/competitions/biohub-cell-tracking-during-development/train",
    )
    parser.add_argument("--limit", type=int, default=5)
    return parser


def _division_count(graph) -> int:
    if hasattr(graph, "node_ids") and hasattr(graph, "out_degree"):
        node_ids = graph.node_ids()
        return sum(1 for degree in graph.out_degree(node_ids) if degree == 2)
    return len(extract_divisions(graph))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    geff_paths = sorted(args.train_root.glob("*.geff"))[: args.limit]
    if not geff_paths:
        print(f"no GEFF stores found under {args.train_root}")
        return 1

    print(f"tracking_cellmot_available={open_dataset is not None}")
    failures = 0
    rows: list[dict] = []
    for geff_path in geff_paths:
        dataset = geff_path.stem
        print(f"dataset={dataset}")
        try:
            if open_dataset is not None:
                ds = open_dataset(
                    geff_path.parent / dataset,
                    normalize=False,
                    require_tracks=True,
                    load_image=False,
                )
                graph = getattr(ds, "tracks", None)
                scale = tuple(getattr(ds, "scale", (1.625, 0.40625, 0.40625)))
                image_shape = getattr(ds, "image_shape", None)
                if graph is None:
                    graph = read_geff_graph(geff_path)
                t_true = read_estimated_number_of_nodes(geff_path)
                if evaluate is not None and per_sample_metrics is not None and node_recall is not None:
                    er = evaluate(graph, graph, scale=scale)
                    recall = node_recall(graph, graph)
                    row = per_sample_metrics(er, float("nan") if t_true is None else t_true, recall)
                    rows.append(row)
                    print(f"  scale={scale}")
                    print(f"  image_shape={image_shape}")
                    print(f"  nodes={er.num_pred_nodes}")
                    print(f"  edges={getattr(graph, 'num_edges')()}")
                    print(f"  divisions={_division_count(graph)}")
                    print(f"  estimated_number_of_nodes={t_true}")
                    print(f"  edge_tp/fp/fn={er.edge_tp}/{er.edge_fp}/{er.edge_fn}")
                    print(f"  division_tp/fp/fn={er.division_tp}/{er.division_fp}/{er.division_fn}")
                    print(f"  node_recall={row['node_recall']}")
                    print(f"  edge_jaccard={row['edge_jaccard']}")
                    print(f"  adj_edge_jaccard={row['adj_edge_jaccard']}")
                    print(f"  total_node_ratio={row['total_node_ratio']}")
                    continue
            else:
                graph = read_geff_graph(geff_path)
                scale = (1.625, 0.40625, 0.40625)
                image_shape = None
            t_true = read_estimated_number_of_nodes(geff_path)
            result = evaluate_one_dataset(graph, graph, scale=scale, t_true=t_true)
        except (OSError, ValueError, FileNotFoundError) as exc:
            failures += 1
            print(f"  ERROR: {exc}")
            continue

        print(f"  scale={scale}")
        print(f"  image_shape={image_shape}")
        print(f"  nodes={len(graph.nodes)}")
        print(f"  edges={len(graph.edges)}")
        print(f"  divisions={len(extract_divisions(graph))}")
        print(f"  estimated_number_of_nodes={t_true}")
        print(f"  edge_tp/fp/fn={result.edge_true_positive}/{result.edge_false_positive}/{result.edge_false_negative}")
        print(
            "  division_tp/fp/fn="
            f"{result.division_true_positive}/{result.division_false_positive}/{result.division_false_negative}"
        )
        print(f"  node_recall={1.0}")
        print(f"  edge_jaccard={result.edge_jaccard:.6f}")
        print(f"  adjusted_edge_jaccard={result.adjusted_edge_jaccard}")
        print(f"  total_node_ratio={result.total_node_ratio}")
        print(f"  division_jaccard={result.division_jaccard:.6f}")
        print(f"  score={result.final_score}")

    if summarise is not None and rows:
        print(f"summary={summarise(rows)}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
