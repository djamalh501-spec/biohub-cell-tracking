"""Create a lightweight CPU baseline submission for Biohub cell tracking."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.evaluation.geff import read_geff_graph
from src.evaluation.official_metric import evaluate_datasets_official_style, read_estimated_number_of_nodes
from src.evaluation.schema import DEFAULT_VOXEL_SPACING_UM, CellNode, SequenceGraph, TemporalEdge
from src.evaluation.validator import OFFICIAL_SUBMISSION_COLUMNS, regenerate_official_row_ids, validate_submission

LOGGER = logging.getLogger(__name__)
DEFAULT_SCALE_ZYX = DEFAULT_VOXEL_SPACING_UM


@dataclass(frozen=True, slots=True)
class ZarrImageInfo:
    dataset: str
    store_path: Path
    array_path: str
    shape: tuple[int, int, int, int]
    scale_zyx_um: tuple[float, float, float]
    uses_npy_fixture: bool = False


@dataclass(frozen=True, slots=True)
class Detection:
    node_id: int
    t: int
    z: int
    y: int
    x: int
    intensity: float


def discover_zarr_datasets(root: str | Path) -> list[Path]:
    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(f"dataset root does not exist: {path}")
    return sorted(item for item in path.glob("*.zarr") if item.is_dir())


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _shape_from_metadata(array_dir: Path) -> tuple[int, ...] | None:
    for name in ("zarr.json", ".zarray"):
        metadata_path = array_dir / name
        if not metadata_path.exists():
            continue
        metadata = _load_json(metadata_path)
        shape = metadata.get("shape")
        if shape is not None:
            return tuple(int(value) for value in shape)
    return None


def _fixture_array_path(store: Path) -> Path | None:
    for candidate in (store / "volume.npy", store / "0.npy", store / "0" / "data.npy"):
        if candidate.exists():
            return candidate
    return None


def _scale_from_multiscales(attrs: dict) -> tuple[float, float, float] | None:
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        return None
    datasets = multiscales[0].get("datasets") if isinstance(multiscales[0], dict) else None
    if not isinstance(datasets, list) or not datasets:
        return None
    transforms = datasets[0].get("coordinateTransformations", [])
    for transform in transforms:
        if not isinstance(transform, dict) or transform.get("type") != "scale":
            continue
        scale = transform.get("scale")
        if isinstance(scale, list) and len(scale) >= 4:
            return (float(scale[-3]), float(scale[-2]), float(scale[-1]))
    return None


def _open_zarr_group(store: Path):
    try:
        import zarr  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "zarr is required to read official image stores. "
            "Smoke tests may use volume.npy fixtures inside .zarr directories."
        ) from exc
    return zarr.open_group(str(store), mode="r")


def _walk_zarr_arrays(group, prefix: str = "") -> list[tuple[str, tuple[int, ...]]]:
    arrays: list[tuple[str, tuple[int, ...]]] = []
    for key in sorted(group.array_keys()):
        array = group[key]
        path = f"{prefix}/{key}" if prefix else str(key)
        arrays.append((path, tuple(int(value) for value in array.shape)))
    for key in sorted(group.group_keys()):
        child = group[key]
        path = f"{prefix}/{key}" if prefix else str(key)
        arrays.extend(_walk_zarr_arrays(child, path))
    return arrays


def inspect_zarr_image(path: str | Path) -> ZarrImageInfo:
    store = Path(path)
    fixture = _fixture_array_path(store)
    if fixture is not None:
        array = np.load(fixture, mmap_mode="r", allow_pickle=False)
        if array.ndim != 4:
            raise ValueError(f"fixture array must be rank 4 (T,Z,Y,X), got shape {array.shape}: {fixture}")
        return ZarrImageInfo(store.stem, store, str(fixture.relative_to(store)), tuple(map(int, array.shape)), DEFAULT_SCALE_ZYX, True)

    try:
        group = _open_zarr_group(store)
    except RuntimeError:
        metadata_candidates = []
        for metadata_path in store.rglob(".zarray"):
            shape = _shape_from_metadata(metadata_path.parent)
            if shape is not None and len(shape) == 4:
                metadata_candidates.append((str(metadata_path.parent.relative_to(store)).replace("\\", "/"), shape))
        for metadata_path in store.rglob("zarr.json"):
            if metadata_path == store / "zarr.json":
                continue
            shape = _shape_from_metadata(metadata_path.parent)
            if shape is not None and len(shape) == 4:
                metadata_candidates.append((str(metadata_path.parent.relative_to(store)).replace("\\", "/"), shape))
        if not metadata_candidates:
            raise
        array_path, shape = sorted(metadata_candidates, key=lambda item: item[0])[0]
        return ZarrImageInfo(store.stem, store, array_path, tuple(map(int, shape)), DEFAULT_SCALE_ZYX, False)

    attrs = dict(getattr(group, "attrs", {}) or {})
    scale = _scale_from_multiscales(attrs) or DEFAULT_SCALE_ZYX
    multiscales = attrs.get("multiscales")
    if isinstance(multiscales, list) and multiscales:
        datasets = multiscales[0].get("datasets") if isinstance(multiscales[0], dict) else None
        if isinstance(datasets, list) and datasets:
            candidate_path = str(datasets[0].get("path", "0"))
            array = group[candidate_path]
            shape = tuple(int(value) for value in array.shape)
            if len(shape) != 4:
                raise ValueError(f"full-resolution OME-Zarr array must be rank 4, got {shape}: {store}/{candidate_path}")
            return ZarrImageInfo(store.stem, store, candidate_path, shape, scale, False)

    arrays = [(array_path, shape) for array_path, shape in _walk_zarr_arrays(group) if len(shape) == 4]
    if not arrays:
        raise ValueError(f"no rank-4 image array found in {store}")
    array_path, shape = sorted(arrays, key=lambda item: (item[0] != "0", item[0]))[0]
    return ZarrImageInfo(store.stem, store, array_path, shape, scale, False)


def load_timepoint(info: ZarrImageInfo, t: int) -> np.ndarray:
    if info.uses_npy_fixture:
        array = np.load(info.store_path / info.array_path, mmap_mode="r", allow_pickle=False)
        return np.asarray(array[t])
    group = _open_zarr_group(info.store_path)
    return np.asarray(group[info.array_path][t])


def robust_normalize(frame: np.ndarray) -> np.ndarray:
    data = np.asarray(frame, dtype=np.float32)
    lo, hi = np.percentile(data, [1.0, 99.0])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        lo = float(np.min(data))
        hi = float(np.max(data))
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)


def _smooth(frame: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return frame
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("scipy is required for --sigma > 0 in classical mode") from exc
    return gaussian_filter(frame, sigma=float(sigma))


def detect_frame_peaks(
    frame: np.ndarray,
    *,
    t: int,
    next_node_id: int,
    threshold_percentile: float,
    min_distance_xy: int,
    min_distance_z: int,
    max_detections: int,
    sigma: float,
) -> tuple[list[Detection], int]:
    image = _smooth(robust_normalize(frame), sigma)
    threshold = float(np.percentile(image, threshold_percentile))
    candidate_coords = np.argwhere(image >= threshold)
    ranked = sorted(
        ((float(image[tuple(coord)]), int(coord[0]), int(coord[1]), int(coord[2])) for coord in candidate_coords),
        key=lambda item: (-item[0], item[1], item[2], item[3]),
    )
    kept: list[Detection] = []
    min_distance_xy = max(0, int(min_distance_xy))
    min_distance_z = max(0, int(min_distance_z))
    for intensity, z, y, x in ranked:
        if intensity <= 0:
            continue
        too_close = any(
            abs(z - det.z) <= min_distance_z and abs(y - det.y) <= min_distance_xy and abs(x - det.x) <= min_distance_xy
            for det in kept
        )
        if too_close:
            continue
        kept.append(Detection(next_node_id, int(t), int(z), int(y), int(x), intensity))
        next_node_id += 1
        if len(kept) >= max_detections:
            break
    return kept, next_node_id


def _distance_matrix_um(
    parents: Sequence[Detection],
    children: Sequence[Detection],
    scale_zyx_um: tuple[float, float, float],
) -> np.ndarray:
    matrix = np.empty((len(parents), len(children)), dtype=float)
    scale = np.asarray(scale_zyx_um, dtype=float)
    for i, parent in enumerate(parents):
        p = np.asarray([parent.z, parent.y, parent.x], dtype=float) * scale
        for j, child in enumerate(children):
            c = np.asarray([child.z, child.y, child.x], dtype=float) * scale
            matrix[i, j] = float(np.linalg.norm(c - p))
    return matrix


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        if max(cost.shape, default=0) > 8:
            raise RuntimeError("scipy is required for one-to-one linking with more than 8 detections per frame")
        return _bruteforce_assignment(cost)
    return linear_sum_assignment(cost)


def _bruteforce_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import itertools

    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    transposed = False
    work = cost
    if n_rows > n_cols:
        work = cost.T
        n_rows, n_cols = work.shape
        transposed = True
    best_cols: tuple[int, ...] | None = None
    best_cost = math.inf
    for cols in itertools.permutations(range(n_cols), n_rows):
        value = float(sum(work[row, col] for row, col in enumerate(cols)))
        if value < best_cost:
            best_cost = value
            best_cols = cols
    rows = np.arange(n_rows, dtype=int)
    cols = np.asarray(best_cols or (), dtype=int)
    return (cols, rows) if transposed else (rows, cols)


def link_detections_one_to_one(
    detections_by_t: dict[int, list[Detection]],
    *,
    scale_zyx_um: tuple[float, float, float],
    link_max_um: float,
) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for t in sorted(detections_by_t):
        parents = detections_by_t.get(t, [])
        children = detections_by_t.get(t + 1, [])
        if not parents or not children:
            continue
        distances = _distance_matrix_um(parents, children, scale_zyx_um)
        rows, cols = _linear_sum_assignment(distances)
        for row, col in zip(rows, cols):
            if distances[int(row), int(col)] <= link_max_um:
                edges.append((parents[int(row)].node_id, children[int(col)].node_id))
    return edges


def graph_from_detections(dataset: str, detections: Iterable[Detection], edges: Iterable[tuple[int, int]]) -> SequenceGraph:
    nodes = tuple(
        CellNode(str(det.node_id), dataset, int(det.t), float(det.z), float(det.y), float(det.x))
        for det in sorted(detections, key=lambda item: (item.t, item.node_id))
    )
    graph_edges = tuple(TemporalEdge(str(source), str(target)) for source, target in sorted(edges))
    graph = SequenceGraph(dataset, nodes, graph_edges)
    errors = graph.validate()
    if errors:
        raise ValueError(f"baseline graph for {dataset!r} failed validation: {'; '.join(errors)}")
    return graph


def graphs_to_submission(graphs: SequenceGraph | Iterable[SequenceGraph]) -> pd.DataFrame:
    graph_list = [graphs] if isinstance(graphs, SequenceGraph) else list(graphs)
    rows: list[dict[str, object]] = []
    for graph in sorted(graph_list, key=lambda item: item.sequence_id):
        for node in sorted(graph.nodes, key=lambda item: (item.time, int(item.node_id))):
            rows.append(
                {
                    "id": -1,
                    "dataset": graph.sequence_id,
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
                    "dataset": graph.sequence_id,
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
    return regenerate_official_row_ids(pd.DataFrame(rows, columns=OFFICIAL_SUBMISSION_COLUMNS))


def make_smoke_graph(info: ZarrImageInfo) -> SequenceGraph:
    t_size, z_size, y_size, x_size = info.shape
    z = max(0, z_size // 2)
    y = max(0, y_size // 2)
    x = max(0, x_size // 2)
    detections = [Detection(1, 0, z, y, x, 1.0)]
    edges: list[tuple[int, int]] = []
    if t_size > 1:
        detections.append(Detection(2, 1, z, y, x, 1.0))
        edges.append((1, 2))
    return graph_from_detections(info.dataset, detections, edges)


def make_classical_graph(info: ZarrImageInfo, args: argparse.Namespace) -> SequenceGraph:
    LOGGER.info("dataset=%s array=%s shape=%s scale_zyx_um=%s", info.dataset, info.array_path, info.shape, info.scale_zyx_um)
    detections_by_t: dict[int, list[Detection]] = {}
    next_node_id = 1
    for t in range(info.shape[0]):
        detections, next_node_id = detect_frame_peaks(
            load_timepoint(info, t),
            t=t,
            next_node_id=next_node_id,
            threshold_percentile=args.threshold_percentile,
            min_distance_xy=args.min_distance_xy,
            min_distance_z=args.min_distance_z,
            max_detections=args.max_detections_per_frame,
            sigma=args.sigma,
        )
        detections_by_t[t] = detections
    edges = link_detections_one_to_one(detections_by_t, scale_zyx_um=info.scale_zyx_um, link_max_um=args.link_max_um)
    return graph_from_detections(info.dataset, [det for values in detections_by_t.values() for det in values], edges)


def _limit_paths(paths: list[Path], limit: int | None) -> list[Path]:
    return paths if limit is None else paths[: max(0, limit)]


def write_and_validate_submission(df: pd.DataFrame, output: Path, sample_submission: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    if sample_submission is None:
        return
    result = validate_submission(output, sample_submission)
    if result.errors:
        raise ValueError("generated submission failed validation: " + "; ".join(result.errors))
    LOGGER.info("submission validation passed with %d warnings", len(result.warnings))


def run_smoke(args: argparse.Namespace) -> int:
    paths = discover_zarr_datasets(args.test_root)
    graphs = [make_smoke_graph(inspect_zarr_image(path)) for path in _limit_paths(paths, args.limit)]
    if not graphs:
        raise ValueError(f"no .zarr datasets discovered under {args.test_root}")
    write_and_validate_submission(graphs_to_submission(graphs), args.output, args.sample_submission)
    LOGGER.info("wrote smoke submission for %d datasets: %s", len(graphs), args.output)
    return 0


def run_classical(args: argparse.Namespace) -> int:
    paths = discover_zarr_datasets(args.test_root)
    graphs = [make_classical_graph(inspect_zarr_image(path), args) for path in _limit_paths(paths, args.limit)]
    if not graphs:
        raise ValueError(f"no .zarr datasets discovered under {args.test_root}")
    write_and_validate_submission(graphs_to_submission(graphs), args.output, args.sample_submission)
    LOGGER.info("wrote classical submission for %d datasets: %s", len(graphs), args.output)
    return 0


def run_eval_on_train(args: argparse.Namespace) -> int:
    if args.train_root is None or not args.train_root.exists():
        LOGGER.warning("train root unavailable; skipping eval-on-train: %s", args.train_root)
        return 0
    pairs: list[tuple[SequenceGraph, SequenceGraph]] = []
    t_true_map: dict[str, float] = {}
    for path in _limit_paths(discover_zarr_datasets(args.train_root), args.limit):
        geff_path = path.with_suffix(".geff")
        if not geff_path.exists():
            LOGGER.warning("missing GEFF for %s; skipping", path.stem)
            continue
        info = inspect_zarr_image(path)
        pred = make_classical_graph(info, args)
        gt = read_geff_graph(geff_path, dataset=path.stem)
        pairs.append((pred, gt))
        t_true = read_estimated_number_of_nodes(geff_path)
        if t_true is not None:
            t_true_map[path.stem] = t_true
    if not pairs:
        LOGGER.warning("no train datasets with GEFF annotations were evaluated")
        return 0
    summary = evaluate_datasets_official_style(pairs, t_true_map=t_true_map)
    for result in summary.per_dataset:
        ratio = result.t_pred / result.t_true if result.t_true else math.nan
        print(
            f"{result.dataset}: edge_jaccard={result.edge_jaccard:.5f} "
            f"adj_edge_jaccard={result.adjusted_edge_jaccard:.5f} "
            f"node_recall=NA T_pred={result.t_pred} T_true={result.t_true} "
            f"T_pred/T_true={ratio:.5f} score={result.final_score:.5f}"
        )
    print(
        f"summary: adj_edge_jaccard={summary.adjusted_edge_jaccard:.5f} "
        f"division_jaccard={summary.division_jaccard:.5f} score={summary.final_score:.5f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Phase 1.1 lightweight Biohub baseline submission.")
    parser.add_argument("--test-root", type=Path, help="Directory containing test/*.zarr stores.")
    parser.add_argument("--train-root", type=Path, help="Directory containing train/*.zarr and train/*.geff stores.")
    parser.add_argument("--sample-submission", type=Path, help="Official sample_submission.csv for validation.")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"), help="Output submission CSV path.")
    parser.add_argument("--mode", choices=("smoke", "classical", "eval-on-train"), required=True)
    parser.add_argument("--threshold-percentile", type=float, default=99.5)
    parser.add_argument("--min-distance-xy", type=int, default=5)
    parser.add_argument("--min-distance-z", type=int, default=2)
    parser.add_argument("--max-detections-per-frame", type=int, default=300)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--link-max-um", type=float, default=5.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    np.random.seed(args.seed)
    try:
        if args.mode in {"smoke", "classical"} and args.test_root is None:
            parser.error(f"--test-root is required for --mode {args.mode}")
        if args.mode == "smoke":
            return run_smoke(args)
        if args.mode == "classical":
            return run_classical(args)
        return run_eval_on_train(args)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
