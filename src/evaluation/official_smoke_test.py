"""Optional official-style smoke test for real Kaggle training GEFF datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .geff import extract_divisions, read_geff_graph
from .official_metric import evaluate_one_dataset, read_estimated_number_of_nodes


try:  # pragma: no cover - optional official package.
    from tracking_cellmot.io import open_dataset  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - expected locally.
    open_dataset = None


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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    geff_paths = sorted(args.train_root.glob("*.geff"))[: args.limit]
    if not geff_paths:
        print(f"no GEFF stores found under {args.train_root}")
        return 1

    print(f"tracking_cellmot_available={open_dataset is not None}")
    failures = 0
    for geff_path in geff_paths:
        dataset = geff_path.stem
        print(f"dataset={dataset}")
        try:
            if open_dataset is not None:
                try:
                    ds = open_dataset(geff_path.parent / f"{dataset}.zarr", geff_path, load_image=False)
                except TypeError:
                    ds = open_dataset(geff_path.parent / f"{dataset}.zarr", geff_path)
                graph = getattr(ds, "tracks", None)
                scale = tuple(getattr(ds, "scale", (1.625, 0.40625, 0.40625)))
                if graph is None:
                    graph = read_geff_graph(geff_path)
            else:
                graph = read_geff_graph(geff_path)
                scale = (1.625, 0.40625, 0.40625)
            t_true = read_estimated_number_of_nodes(geff_path)
            result = evaluate_one_dataset(graph, graph, scale=scale, t_true=t_true)
        except (OSError, ValueError, FileNotFoundError) as exc:
            failures += 1
            print(f"  ERROR: {exc}")
            continue

        print(f"  nodes={len(graph.nodes)}")
        print(f"  edges={len(graph.edges)}")
        print(f"  divisions={len(extract_divisions(graph))}")
        print(f"  estimated_number_of_nodes={t_true}")
        print(f"  edge_jaccard={result.edge_jaccard:.6f}")
        print(f"  adjusted_edge_jaccard={result.adjusted_edge_jaccard}")
        print(f"  division_jaccard={result.division_jaccard:.6f}")
        print(f"  score={result.final_score}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
