"""Official-style metric bridge with guarded optional tracking_cellmot support."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Mapping, Sequence
import warnings

from .geff import extract_divisions
from .matching import match_nodes
from .schema import DEFAULT_VOXEL_SPACING_UM, SequenceGraph

LOGGER = logging.getLogger(__name__)


try:  # pragma: no cover - optional official package.
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - expected locally.
    evaluate = None
    node_recall = None
    per_sample_metrics = None
    summarise = None


@dataclass(frozen=True, slots=True)
class OfficialStyleMetricResult:
    dataset: str
    edge_true_positive: int
    edge_false_positive: int
    edge_false_negative: int
    edge_jaccard: float
    adjusted_edge_jaccard: float
    division_true_positive: int
    division_false_positive: int
    division_false_negative: int
    division_jaccard: float
    final_score: float
    total_node_ratio: float
    t_pred: int
    t_true: float | None
    edge_weight: int
    tracking_cellmot_used: bool
    fallback_note: str


@dataclass(frozen=True, slots=True)
class OfficialStyleSummary:
    adjusted_edge_jaccard: float
    division_jaccard: float
    final_score: float
    per_dataset: tuple[OfficialStyleMetricResult, ...]
    tracking_cellmot_used: bool


def adjusted_edge_jaccard(edge_jaccard: float, t_pred: int, t_true: float | None) -> float:
    """Apply the official adjusted Edge Jaccard formula."""

    if t_true is None or not math.isfinite(t_true) or t_true <= 0:
        return math.nan
    return max(0.0, edge_jaccard * (1.0 - 0.1 * (t_pred - t_true) / t_true))


def total_node_ratio(t_pred: int, t_true: float | None) -> float:
    """Return official ``(T_pred - T_true) / T_true`` or NaN when unavailable."""

    if t_true is None or not math.isfinite(t_true) or t_true <= 0:
        return math.nan
    return (t_pred - t_true) / t_true


def _edge_counts_official_fallback(pred_graph: SequenceGraph, gt_graph: SequenceGraph, scale) -> tuple[int, int, int]:
    matches = match_nodes(gt_graph.nodes, pred_graph.nodes, voxel_spacing=scale, max_distance_um=7.0)
    gt_edges = {(edge.parent_id, edge.child_id) for edge in gt_graph.edges}
    gt_parents_by_child: dict[str, set[str]] = defaultdict(set)
    gt_children_by_parent: dict[str, set[str]] = defaultdict(set)
    for parent_id, child_id in gt_edges:
        gt_parents_by_child[child_id].add(parent_id)
        gt_children_by_parent[parent_id].add(child_id)

    recovered_gt_edges: set[tuple[str, str]] = set()
    false_positive = 0
    for edge in pred_graph.edges:
        mapped_source = matches.pred_to_gt.get(edge.parent_id)
        mapped_target = matches.pred_to_gt.get(edge.child_id)
        if mapped_source is not None and mapped_target is not None and (mapped_source, mapped_target) in gt_edges:
            recovered_gt_edges.add((mapped_source, mapped_target))
            continue

        conflicts_target_parent = (
            mapped_target is not None
            and mapped_target in gt_parents_by_child
            and mapped_source not in gt_parents_by_child[mapped_target]
        )
        conflicts_source_child = (
            mapped_source is not None
            and mapped_source in gt_children_by_parent
            and mapped_target not in gt_children_by_parent[mapped_source]
        )
        if conflicts_target_parent or conflicts_source_child:
            false_positive += 1

    true_positive = len(recovered_gt_edges)
    false_negative = len(gt_edges - recovered_gt_edges)
    return true_positive, false_positive, false_negative


def _division_counts_fallback(pred_graph: SequenceGraph, gt_graph: SequenceGraph, scale) -> tuple[int, int, int]:
    matches = match_nodes(gt_graph.nodes, pred_graph.nodes, voxel_spacing=scale, max_distance_um=7.0)
    gt_divisions = {
        (division.parent_id, frozenset(division.child_ids))
        for division in extract_divisions(gt_graph)
    }
    mapped_pred_divisions = set()
    false_positive = 0
    for division in extract_divisions(pred_graph):
        parent = matches.pred_to_gt.get(division.parent_id)
        children = [matches.pred_to_gt.get(child_id) for child_id in division.child_ids]
        if parent is None or any(child is None for child in children):
            false_positive += 1
            continue
        mapped_pred_divisions.add((parent, frozenset(child for child in children if child is not None)))
    true_positive = len(gt_divisions & mapped_pred_divisions)
    false_positive += len(mapped_pred_divisions - gt_divisions)
    false_negative = len(gt_divisions - mapped_pred_divisions)
    return true_positive, false_positive, false_negative


def _jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def evaluate_one_dataset(
    pred_graph: SequenceGraph,
    gt_graph: SequenceGraph,
    scale: tuple[float, float, float] = DEFAULT_VOXEL_SPACING_UM,
    t_true: float | None = None,
) -> OfficialStyleMetricResult:
    """Evaluate one dataset with official-style semantics.

    If ``tracking_cellmot`` is unavailable, this uses a dependency-light fallback
    implementing the documented edge counts and adjusted-jaccard formula. The fallback
    division metric is not the official tolerant division metric.
    """

    if evaluate is not None:
        LOGGER.warning("tracking_cellmot is installed, but local SequenceGraph bridge uses fallback counts.")

    edge_tp, edge_fp, edge_fn = _edge_counts_official_fallback(pred_graph, gt_graph, scale)
    edge_j = _jaccard(edge_tp, edge_fp, edge_fn)
    adjusted = adjusted_edge_jaccard(edge_j, len(pred_graph.nodes), t_true)
    ratio = total_node_ratio(len(pred_graph.nodes), t_true)
    div_tp, div_fp, div_fn = _division_counts_fallback(pred_graph, gt_graph, scale)
    div_j = _jaccard(div_tp, div_fp, div_fn)
    final = math.nan if math.isnan(adjusted) else adjusted + (0.1 * div_j if div_tp + div_fp + div_fn > 0 else 0.0)
    if t_true is None:
        LOGGER.warning("T_true is unavailable; adjusted_edge_jaccard and final_score are NaN")
    return OfficialStyleMetricResult(
        dataset=pred_graph.sequence_id,
        edge_true_positive=edge_tp,
        edge_false_positive=edge_fp,
        edge_false_negative=edge_fn,
        edge_jaccard=edge_j,
        adjusted_edge_jaccard=adjusted,
        division_true_positive=div_tp,
        division_false_positive=div_fp,
        division_false_negative=div_fn,
        division_jaccard=div_j,
        final_score=final,
        total_node_ratio=ratio,
        t_pred=len(pred_graph.nodes),
        t_true=t_true,
        edge_weight=edge_tp + edge_fp + edge_fn,
        tracking_cellmot_used=False,
        fallback_note=(
            "Dependency-light fallback. Edge counts follow documented sparse-GT conflict rules; "
            "division counts are exact out-degree matches, not the official tolerant division metric."
        ),
    )


def evaluate_datasets_official_style(
    pairs: Sequence[tuple[SequenceGraph, SequenceGraph]],
    scale: tuple[float, float, float] = DEFAULT_VOXEL_SPACING_UM,
    t_true_map: Mapping[str, float] | None = None,
) -> OfficialStyleSummary:
    results = tuple(
        evaluate_one_dataset(
            pred_graph,
            gt_graph,
            scale=scale,
            t_true=None if t_true_map is None else t_true_map.get(gt_graph.sequence_id),
        )
        for pred_graph, gt_graph in pairs
    )
    total_weight = sum(result.edge_weight for result in results)
    if total_weight == 0 or any(math.isnan(result.adjusted_edge_jaccard) for result in results):
        adjusted = math.nan
    else:
        adjusted = sum(result.adjusted_edge_jaccard * result.edge_weight for result in results) / total_weight
    div_tp = sum(result.division_true_positive for result in results)
    div_fp = sum(result.division_false_positive for result in results)
    div_fn = sum(result.division_false_negative for result in results)
    division = math.nan if div_tp + div_fp + div_fn == 0 else _jaccard(div_tp, div_fp, div_fn)
    if math.isnan(adjusted):
        final = math.nan
    else:
        final = adjusted + (0.1 * division if not math.isnan(division) else 0.0)
    return OfficialStyleSummary(
        adjusted_edge_jaccard=adjusted,
        division_jaccard=division,
        final_score=final,
        per_dataset=results,
        tracking_cellmot_used=False,
    )


def _attrs_from_metadata_file(metadata_path: Path) -> dict:
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if metadata_path.name == "zarr.json" and isinstance(data.get("attributes"), dict):
        return data["attributes"]
    return data if isinstance(data, dict) else {}


def read_estimated_number_of_nodes(geff_path: str | Path) -> float | None:
    """Read official GEFF ``geff.extra.estimated_number_of_nodes`` metadata.

    Supports Zarr v3 ``zarr.json`` attribute storage and Zarr v2 ``.zattrs``.
    The legacy ``extra.estimated_number_of_nodes`` location is retained as a
    deprecated fallback.
    """

    path = Path(geff_path)
    metadata_paths = [path / "zarr.json", path / ".zattrs"]
    for metadata_path in metadata_paths:
        if not metadata_path.exists():
            continue
        attrs = _attrs_from_metadata_file(metadata_path)
        geff_attrs = attrs.get("geff")
        if isinstance(geff_attrs, dict):
            extra = geff_attrs.get("extra")
            if isinstance(extra, dict) and extra.get("estimated_number_of_nodes") is not None:
                return float(extra["estimated_number_of_nodes"])
        legacy_extra = attrs.get("extra")
        if isinstance(legacy_extra, dict) and legacy_extra.get("estimated_number_of_nodes") is not None:
            warnings.warn(
                "GEFF metadata at attrs['extra']['estimated_number_of_nodes'] is deprecated; "
                "expected attrs['geff']['extra']['estimated_number_of_nodes']",
                DeprecationWarning,
                stacklevel=2,
            )
            return float(legacy_extra["estimated_number_of_nodes"])
    message = f"estimated_number_of_nodes not found in GEFF metadata: {path}"
    LOGGER.warning("%s", message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return None
