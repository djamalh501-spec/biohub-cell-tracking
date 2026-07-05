"""Lineage graph metrics built on local node matching."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

from .matching import NodeMatchResult, match_nodes
from .schema import DEFAULT_VOXEL_SPACING_UM, SequenceGraph, TemporalEdge


@dataclass(frozen=True, slots=True)
class EdgeJaccardResult:
    score: float
    raw_score: float
    node_overprediction_penalty: float
    penalty_formula_verified: bool
    penalty_note: str
    true_positive_edges: int
    false_positive_edges: int
    false_negative_edges: int
    intersection_size: int
    union_size: int


@dataclass(frozen=True, slots=True)
class DivisionJaccardResult:
    score: float
    matched_divisions: int
    false_positive_divisions: int
    false_negative_divisions: int
    intersection_size: int
    union_size: int
    assumption: str


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    edge_jaccard: EdgeJaccardResult
    division_jaccard: DivisionJaccardResult
    node_precision: float
    node_recall: float
    matched_nodes: int
    ground_truth_nodes: int
    predicted_nodes: int
    validation_warnings: tuple[str, ...]
    combined_score: None
    combined_score_note: str


def _edge_set(edges: tuple[TemporalEdge, ...]) -> set[tuple[str, str]]:
    return {(edge.parent_id, edge.child_id) for edge in edges}


def _mapped_prediction_edges(
    prediction: SequenceGraph,
    matches: NodeMatchResult,
) -> set[tuple[str, str]]:
    mapped_edges: set[tuple[str, str]] = set()
    for edge in prediction.edges:
        gt_parent = matches.pred_to_gt.get(edge.parent_id)
        gt_child = matches.pred_to_gt.get(edge.child_id)
        if gt_parent is not None and gt_child is not None:
            mapped_edges.add((gt_parent, gt_child))
    return mapped_edges


def raw_edge_jaccard(
    ground_truth: SequenceGraph,
    prediction: SequenceGraph,
    matches: NodeMatchResult,
) -> EdgeJaccardResult:
    """Compute temporal edge Jaccard after mapping predictions to GT IDs.

    Predicted edges touching unmatched prediction nodes cannot be mapped and count as false
    positives through the union term.
    """

    gt_edges = _edge_set(ground_truth.edges)
    mapped_pred_edges = _mapped_prediction_edges(prediction, matches)
    unmappable_pred_edges = len(prediction.edges) - len(mapped_pred_edges)
    intersection = gt_edges & mapped_pred_edges
    union_size = len(gt_edges | mapped_pred_edges) + unmappable_pred_edges
    score = 1.0 if union_size == 0 else len(intersection) / union_size
    false_positive = (len(mapped_pred_edges - gt_edges) + unmappable_pred_edges)
    false_negative = len(gt_edges - mapped_pred_edges)
    return EdgeJaccardResult(
        score=score,
        raw_score=score,
        node_overprediction_penalty=1.0,
        penalty_formula_verified=False,
        penalty_note="Raw Edge Jaccard only; no node overprediction penalty applied.",
        true_positive_edges=len(intersection),
        false_positive_edges=false_positive,
        false_negative_edges=false_negative,
        intersection_size=len(intersection),
        union_size=union_size,
    )


def node_overprediction_penalty_approx(ground_truth_node_count: int, predicted_node_count: int) -> float:
    """Deprecated pre-Phase-0.4 node overprediction approximation.

    Use ``src.evaluation.official_metric.adjusted_edge_jaccard`` for the documented
    official adjusted Edge Jaccard formula. This approximation is retained only for
    backward compatibility with Phase 0.2/0.3 local tests.
    """

    warnings.warn(
        "node_overprediction_penalty_approx is deprecated; use official_metric.adjusted_edge_jaccard",
        DeprecationWarning,
        stacklevel=2,
    )
    if predicted_node_count <= ground_truth_node_count:
        return 1.0
    if predicted_node_count == 0:
        return 1.0
    return ground_truth_node_count / predicted_node_count


def edge_jaccard_approx(
    ground_truth: SequenceGraph,
    prediction: SequenceGraph,
    matches: NodeMatchResult,
) -> EdgeJaccardResult:
    """Deprecated local approximation; not an official competition metric."""

    raw = raw_edge_jaccard(ground_truth, prediction, matches)
    penalty = node_overprediction_penalty_approx(len(ground_truth.nodes), len(prediction.nodes))
    return EdgeJaccardResult(
        score=raw.raw_score * penalty,
        raw_score=raw.raw_score,
        node_overprediction_penalty=penalty,
        penalty_formula_verified=False,
        penalty_note=(
            "Approximation: score = raw Edge Jaccard * min(1, ground_truth_nodes / "
            "predicted_nodes). The official penalty formula was not available locally."
        ),
        true_positive_edges=raw.true_positive_edges,
        false_positive_edges=raw.false_positive_edges,
        false_negative_edges=raw.false_negative_edges,
        intersection_size=raw.intersection_size,
        union_size=raw.union_size,
    )


def edge_jaccard(
    ground_truth: SequenceGraph,
    prediction: SequenceGraph,
    matches: NodeMatchResult,
) -> EdgeJaccardResult:
    """Return deprecated local approximate Edge Jaccard.

    Prefer ``src.evaluation.official_metric.evaluate_one_dataset`` for Phase 0.4
    official-style metrics.
    """

    return edge_jaccard_approx(ground_truth, prediction, matches)


def _division_set(graph: SequenceGraph) -> set[tuple[str, frozenset[str]]]:
    return {(division.parent_id, frozenset(division.child_ids)) for division in graph.divisions}


def _mapped_prediction_divisions(
    prediction: SequenceGraph,
    matches: NodeMatchResult,
) -> tuple[set[tuple[str, frozenset[str]]], int]:
    mapped: set[tuple[str, frozenset[str]]] = set()
    unmappable = 0
    for division in prediction.divisions:
        parent = matches.pred_to_gt.get(division.parent_id)
        children = [matches.pred_to_gt.get(child_id) for child_id in division.child_ids]
        if parent is None or any(child is None for child in children):
            unmappable += 1
            continue
        mapped.add((parent, frozenset(child for child in children if child is not None)))
    return mapped, unmappable


def division_jaccard(
    ground_truth: SequenceGraph,
    prediction: SequenceGraph,
    matches: NodeMatchResult,
) -> DivisionJaccardResult:
    """Compute Division Jaccard using exact matched parent and child-set identity.

    TODO: Verify against official Biohub metric code when available. The current
    assumption is that a division is correct only when the matched parent and the
    unordered set of matched division children are exactly the same as ground truth.
    """

    assumption = (
        "Divisions are compared as exact matched parent plus unordered matched child set; "
        "official tie-breaking or partial-credit behavior is unverified."
    )
    gt_divisions = _division_set(ground_truth)
    mapped_pred_divisions, unmappable = _mapped_prediction_divisions(prediction, matches)
    intersection = gt_divisions & mapped_pred_divisions
    union_size = len(gt_divisions | mapped_pred_divisions) + unmappable
    score = 1.0 if union_size == 0 else len(intersection) / union_size
    return DivisionJaccardResult(
        score=score,
        matched_divisions=len(intersection),
        false_positive_divisions=len(mapped_pred_divisions - gt_divisions) + unmappable,
        false_negative_divisions=len(gt_divisions - mapped_pred_divisions),
        intersection_size=len(intersection),
        union_size=union_size,
        assumption=assumption,
    )


def evaluate_lineage_graph(
    ground_truth: SequenceGraph,
    prediction: SequenceGraph,
    *,
    voxel_spacing: tuple[float, float, float] = DEFAULT_VOXEL_SPACING_UM,
    matching_threshold_um: float = 7.0,
) -> EvaluationResult:
    """Run validation, node matching, Edge Jaccard, and Division Jaccard."""

    validation_warnings = tuple(
        f"ground_truth: {error}" for error in ground_truth.validate()
    ) + tuple(f"prediction: {error}" for error in prediction.validate())
    matches = match_nodes(
        ground_truth.nodes,
        prediction.nodes,
        voxel_spacing=voxel_spacing,
        max_distance_um=matching_threshold_um,
    )
    edge_result = edge_jaccard_approx(ground_truth, prediction, matches)
    division_result = division_jaccard(ground_truth, prediction, matches)
    matched_nodes = len(matches.gt_to_pred)
    node_precision = 1.0 if not prediction.nodes else matched_nodes / len(prediction.nodes)
    node_recall = 1.0 if not ground_truth.nodes else matched_nodes / len(ground_truth.nodes)
    return EvaluationResult(
        edge_jaccard=edge_result,
        division_jaccard=division_result,
        node_precision=node_precision,
        node_recall=node_recall,
        matched_nodes=matched_nodes,
        ground_truth_nodes=len(ground_truth.nodes),
        predicted_nodes=len(prediction.nodes),
        validation_warnings=validation_warnings,
        combined_score=None,
        combined_score_note=(
            "No combined score is returned because the official weighting between Edge "
            "Jaccard and Division Jaccard has not been verified locally."
        ),
    )
