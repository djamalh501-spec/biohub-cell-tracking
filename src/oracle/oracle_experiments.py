"""Oracle perturbation experiments for the Biohub tracking metric.

These experiments perturb canonical submission DataFrames and evaluate them against
training annotations. They are not modeling code and never load full image volumes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.geff import extract_divisions
from src.evaluation.official_bridge import submission_df_to_tracksdata, tracksdata_to_submission_df
from src.evaluation.official_metric import (
    adjusted_edge_jaccard,
    evaluate_one_dataset,
    read_estimated_number_of_nodes,
    total_node_ratio,
)
from src.evaluation.schema import SequenceGraph
from src.evaluation.validator import regenerate_official_row_ids


try:  # pragma: no cover - optional Kaggle dependency path.
    from tracking_cellmot.io import open_dataset  # type: ignore[import-not-found]
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - local default.
    open_dataset = None
    evaluate = None
    node_recall = None
    per_sample_metrics = None
    summarise = None


NODE_DROPOUT_P = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
COORD_JITTER_SIGMA_UM = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
NODE_COUNT_RATIOS = (0.001, 0.01, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 5.0, 20.0)
EDGE_SWAP_P = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
DEFAULT_SCALE = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    dataset: str
    graph: object
    submission_df: pd.DataFrame
    scale: tuple[float, float, float]
    image_shape: tuple[int, ...] | None
    t_true: float | None


def _node_mask(df: pd.DataFrame) -> pd.Series:
    return df["row_type"].eq("node")


def _edge_mask(df: pd.DataFrame) -> pd.Series:
    return df["row_type"].eq("edge")


def _copy_regenerate(df: pd.DataFrame) -> pd.DataFrame:
    return regenerate_official_row_ids(df.reset_index(drop=True))


def node_dropout(df: pd.DataFrame, p: float, rng: np.random.Generator) -> pd.DataFrame:
    """Remove a fraction of node rows and all incident edges."""

    if p <= 0:
        return _copy_regenerate(df.copy())
    output = df.copy()
    node_rows = output[_node_mask(output)]
    drop_keys: set[tuple[str, int]] = set()
    for row in node_rows.itertuples(index=False):
        if rng.random() < p:
            drop_keys.add((str(row.dataset), int(row.node_id)))
    if not drop_keys:
        return _copy_regenerate(output)

    keep = []
    for row in output.itertuples(index=False):
        dataset = str(row.dataset)
        if row.row_type == "node":
            keep.append((dataset, int(row.node_id)) not in drop_keys)
        else:
            keep.append(
                (dataset, int(row.source_id)) not in drop_keys
                and (dataset, int(row.target_id)) not in drop_keys
            )
    return _copy_regenerate(output.loc[keep])


def coord_jitter(
    df: pd.DataFrame,
    sigma_um: float,
    rng: np.random.Generator,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    image_shape: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Add Gaussian coordinate noise in physical micrometres and return voxel rows."""

    output = df.copy()
    if sigma_um <= 0:
        return _copy_regenerate(output)
    node_indices = output.index[_node_mask(output)]
    voxel_sigma = np.asarray([sigma_um / scale[0], sigma_um / scale[1], sigma_um / scale[2]], dtype=float)
    noise = rng.normal(0.0, voxel_sigma, size=(len(node_indices), 3))
    coords = output.loc[node_indices, ["z", "y", "x"]].to_numpy(dtype=float) + noise
    if image_shape is not None and len(image_shape) >= 4:
        bounds = np.asarray([image_shape[-3] - 1, image_shape[-2] - 1, image_shape[-1] - 1], dtype=float)
        coords = np.clip(coords, 0, bounds)
    output.loc[node_indices, ["z", "y", "x"]] = np.rint(coords).astype(int)
    return _copy_regenerate(output)


def node_count_adjustment_curve(num_pred_nodes: int, ratios: Iterable[float] = NODE_COUNT_RATIOS) -> pd.DataFrame:
    """Isolate the official T_pred/T_true adjusted-jaccard multiplier."""

    rows = []
    for ratio in ratios:
        effective_t_true = num_pred_nodes / ratio
        adj = adjusted_edge_jaccard(1.0, num_pred_nodes, effective_t_true)
        rows.append(
            {
                "ratio": ratio,
                "effective_t_true": effective_t_true,
                "total_node_ratio": total_node_ratio(num_pred_nodes, effective_t_true),
                "edge_jaccard": 1.0,
                "adj_edge_jaccard": adj,
                "score": adj,
            }
        )
    return pd.DataFrame(rows)


def division_removal(df: pd.DataFrame) -> pd.DataFrame:
    """Remove one daughter edge from every two-child parent."""

    output = df.copy()
    edge_rows = output[_edge_mask(output)]
    remove_indices: list[int] = []
    for (dataset, source_id), group in edge_rows.groupby(["dataset", "source_id"], sort=True):
        unique_targets = sorted(set(int(value) for value in group["target_id"]))
        if len(unique_targets) == 2:
            target_to_remove = unique_targets[-1]
            matching = group[group["target_id"].astype(int).eq(target_to_remove)]
            remove_indices.append(int(matching.index[0]))
    return _copy_regenerate(output.drop(index=remove_indices))


def edge_swap(df: pd.DataFrame, p: float, rng: np.random.Generator) -> pd.DataFrame:
    """Rewire a fraction of edges to incorrect forward-time targets."""

    output = df.copy()
    if p <= 0:
        return _copy_regenerate(output)
    node_rows = output[_node_mask(output)]
    node_times = {
        (str(row.dataset), int(row.node_id)): int(row.t)
        for row in node_rows.itertuples(index=False)
    }
    nodes_by_dataset = {
        dataset: group.copy()
        for dataset, group in node_rows.groupby("dataset", sort=False)
    }
    existing_edges = {
        (str(row.dataset), int(row.source_id), int(row.target_id))
        for row in output[_edge_mask(output)].itertuples(index=False)
    }

    for index, row in output[_edge_mask(output)].iterrows():
        if rng.random() >= p:
            continue
        dataset = str(row["dataset"])
        source_id = int(row["source_id"])
        target_id = int(row["target_id"])
        source_t = node_times.get((dataset, source_id))
        if source_t is None:
            continue
        candidates = nodes_by_dataset[dataset]
        original_target_t = node_times.get((dataset, target_id))
        preferred = candidates[
            candidates["t"].astype(int).gt(source_t)
            & ~candidates["node_id"].astype(int).isin([source_id, target_id])
            & candidates["t"].astype(int).eq(original_target_t)
        ]
        if preferred.empty:
            preferred = candidates[
                candidates["t"].astype(int).gt(source_t)
                & ~candidates["node_id"].astype(int).isin([source_id, target_id])
            ]
        candidate_ids = [int(value) for value in preferred["node_id"].tolist()]
        rng.shuffle(candidate_ids)
        replacement = None
        for candidate_id in candidate_ids:
            if (dataset, source_id, candidate_id) not in existing_edges:
                replacement = candidate_id
                break
        if replacement is None:
            continue
        existing_edges.discard((dataset, source_id, target_id))
        existing_edges.add((dataset, source_id, replacement))
        output.at[index, "target_id"] = replacement
    return _copy_regenerate(output)


def count_divisions_in_df(df: pd.DataFrame) -> int:
    edge_rows = df[_edge_mask(df)]
    return sum(1 for _, group in edge_rows.groupby(["dataset", "source_id"]) if group["target_id"].nunique() == 2)


def _evaluate_prediction(pred_df: pd.DataFrame, loaded: LoadedDataset) -> dict[str, float | int | str]:
    pred_graph = submission_df_to_tracksdata(pred_df)[loaded.dataset]
    if evaluate is not None and per_sample_metrics is not None and node_recall is not None:
        er = evaluate(pred_graph, loaded.graph, scale=loaded.scale)
        row = per_sample_metrics(er, float("nan") if loaded.t_true is None else loaded.t_true, node_recall(pred_graph, loaded.graph))
        return {
            "dataset": loaded.dataset,
            **row,
            "score": _score_from_row(row),
        }
    result = evaluate_one_dataset(pred_graph, loaded.graph, scale=loaded.scale, t_true=loaded.t_true)
    return {
        "dataset": loaded.dataset,
        "edge_tp": result.edge_true_positive,
        "edge_fp": result.edge_false_positive,
        "edge_fn": result.edge_false_negative,
        "division_tp": result.division_true_positive,
        "division_fp": result.division_false_positive,
        "division_fn": result.division_false_negative,
        "node_recall": 1.0,
        "total_node_ratio": result.total_node_ratio,
        "edge_jaccard": result.edge_jaccard,
        "adj_edge_jaccard": result.adjusted_edge_jaccard,
        "score": result.final_score,
    }


def _score_from_row(row: dict) -> float:
    if row["adj_edge_jaccard"] != row["adj_edge_jaccard"]:
        return float("nan")
    div_total = row["division_tp"] + row["division_fp"] + row["division_fn"]
    if div_total == 0:
        return row["adj_edge_jaccard"]
    return row["adj_edge_jaccard"] + 0.1 * (row["division_tp"] / div_total)


def _load_datasets(train_root: Path, limit: int, prefer_divisions: bool) -> list[LoadedDataset]:
    if open_dataset is None:
        raise RuntimeError("tracking_cellmot is unavailable; oracle CLI requires official dependencies")
    paths = sorted(train_root.glob("*.geff"))
    if prefer_divisions:
        selected = []
        for path in paths:
            ds = open_dataset(path.parent / path.stem, normalize=False, require_tracks=True, load_image=False)
            df = tracksdata_to_submission_df({path.stem: ds.tracks})
            if count_divisions_in_df(df) > 0:
                selected.append((path, ds, df))
            if len(selected) >= limit:
                break
    else:
        selected = []
        for path in paths[:limit]:
            ds = open_dataset(path.parent / path.stem, normalize=False, require_tracks=True, load_image=False)
            df = tracksdata_to_submission_df({path.stem: ds.tracks})
            selected.append((path, ds, df))
    loaded = []
    for path, ds, df in selected:
        loaded.append(
            LoadedDataset(
                dataset=path.stem,
                graph=ds.tracks,
                submission_df=df,
                scale=tuple(ds.scale),
                image_shape=tuple(ds.image_shape) if ds.image_shape is not None else None,
                t_true=read_estimated_number_of_nodes(path),
            )
        )
    return loaded


def _write_report(output_dir: Path, datasets: list[LoadedDataset], csv_paths: list[Path]) -> None:
    report = [
        "# Oracle Findings",
        "",
        "No modeling, detector, GPU, or image loading was used.",
        "",
        "## Datasets",
        "",
    ]
    for loaded in datasets:
        report.append(
            f"- `{loaded.dataset}`: nodes={len(loaded.submission_df[loaded.submission_df['row_type'] == 'node'])}, "
            f"edges={len(loaded.submission_df[loaded.submission_df['row_type'] == 'edge'])}, "
            f"divisions={count_divisions_in_df(loaded.submission_df)}, T_true={loaded.t_true}"
        )
    report.extend(
        [
            "",
            "## Outputs",
            "",
            *[f"- `{path}`" for path in csv_paths],
            "",
            "## Interpretation",
            "",
            "- Node recall, spatial accuracy near 7 um, and edge correctness should be treated as primary levers.",
            "- Node-count calibration only enters through the flat T_pred adjustment.",
            "- Divisions are rare in the verified sparse labels but still affect the final score when present.",
            "- Sparse-GT self-evaluation above 1.0 is a ceiling artifact, not a progress signal.",
        ]
    )
    (output_dir / "ORACLE_FINDINGS.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_oracle_experiments(train_root: Path, limit: int, seed: int, output_dir: Path, prefer_divisions: bool) -> list[Path]:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _load_datasets(train_root, limit=limit, prefer_divisions=prefer_divisions)
    if not datasets:
        raise RuntimeError("no datasets selected")

    written: list[Path] = []

    rows = []
    for loaded in datasets:
        for p in NODE_DROPOUT_P:
            rows.append({"experiment": "node_dropout", "p": p, **_evaluate_prediction(node_dropout(loaded.submission_df, p, rng), loaded)})
    path = output_dir / "node_dropout.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    written.append(path)

    rows = []
    for loaded in datasets:
        for sigma in COORD_JITTER_SIGMA_UM:
            rows.append(
                {
                    "experiment": "coord_jitter",
                    "sigma_um": sigma,
                    **_evaluate_prediction(coord_jitter(loaded.submission_df, sigma, rng, loaded.scale, loaded.image_shape), loaded),
                }
            )
    path = output_dir / "coord_jitter.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    written.append(path)

    rows = []
    for loaded in datasets:
        num_pred_nodes = int(_node_mask(loaded.submission_df).sum())
        curve = node_count_adjustment_curve(num_pred_nodes)
        curve.insert(0, "dataset", loaded.dataset)
        rows.append(curve)
    path = output_dir / "node_count_adjustment_curve.csv"
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    written.append(path)

    rows = []
    for loaded in datasets:
        perfect = _evaluate_prediction(loaded.submission_df, loaded)
        perturbed = _evaluate_prediction(division_removal(loaded.submission_df), loaded)
        rows.append({"experiment": "division_removal", "score_delta": perturbed["score"] - perfect["score"], **perturbed})
    path = output_dir / "division_removal.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    written.append(path)

    rows = []
    for loaded in datasets:
        for p in EDGE_SWAP_P:
            rows.append({"experiment": "edge_swap", "p": p, **_evaluate_prediction(edge_swap(loaded.submission_df, p, rng), loaded)})
    path = output_dir / "edge_swap.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    written.append(path)

    _write_report(output_dir, datasets, written)
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run oracle metric-sensitivity experiments.")
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oracle"))
    parser.add_argument("--prefer-divisions", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.train_root.exists():
        print(f"train root does not exist: {args.train_root}")
        return 1
    if open_dataset is None:
        print("tracking_cellmot is unavailable; skipping oracle experiments")
        return 1
    paths = run_oracle_experiments(
        train_root=args.train_root,
        limit=args.limit,
        seed=args.seed,
        output_dir=args.output_dir,
        prefer_divisions=args.prefer_divisions,
    )
    print(f"output CSV paths: {[str(path) for path in paths]}")
    for path in paths:
        print(f"\n{path}")
        print(pd.read_csv(path).head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
