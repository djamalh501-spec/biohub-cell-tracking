"""Physical coordinate conversion and one-to-one node matching."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np

try:  # pragma: no cover - exercised only when scipy is installed.
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except ModuleNotFoundError:  # pragma: no cover - fallback is covered instead.
    _scipy_linear_sum_assignment = None

from .schema import CellNode, DEFAULT_VOXEL_SPACING_UM


@dataclass(frozen=True, slots=True)
class NodeMatchResult:
    """One-to-one node matching result."""

    gt_to_pred: dict[str, str]
    pred_to_gt: dict[str, str]
    distances_um: dict[tuple[str, str], float]
    unmatched_gt: set[str]
    unmatched_pred: set[str]


def voxel_to_physical_um(
    z: float,
    y: float,
    x: float,
    voxel_spacing: tuple[float, float, float] = DEFAULT_VOXEL_SPACING_UM,
) -> tuple[float, float, float]:
    """Convert voxel ``(z, y, x)`` coordinates to physical micrometres."""

    z_spacing, y_spacing, x_spacing = voxel_spacing
    return (z * z_spacing, y * y_spacing, x * x_spacing)


def _node_key(node: CellNode) -> tuple[str, int]:
    return (node.sequence_id, node.time)


def _physical_array(nodes: list[CellNode], voxel_spacing: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(
        [voxel_to_physical_um(node.z, node.y, node.x, voxel_spacing) for node in nodes],
        dtype=float,
    )


def _fallback_linear_sum_assignment(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact rectangular assignment fallback for environments without SciPy."""

    rows, columns = costs.shape
    transposed = False
    working = costs
    if rows > columns:
        working = costs.T
        rows, columns = working.shape
        transposed = True

    memo: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}

    def solve(row: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if row == rows:
            return 0.0, ()
        key = (row, used_mask)
        if key in memo:
            return memo[key]
        best_cost = math.inf
        best_assignment: tuple[int, ...] = ()
        for column in range(columns):
            if used_mask & (1 << column):
                continue
            tail_cost, tail_assignment = solve(row + 1, used_mask | (1 << column))
            total_cost = float(working[row, column]) + tail_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_assignment = (column,) + tail_assignment
        memo[key] = (best_cost, best_assignment)
        return memo[key]

    _, assignment = solve(0, 0)
    row_indices = np.arange(rows, dtype=int)
    column_indices = np.asarray(assignment, dtype=int)
    if transposed:
        return column_indices, row_indices
    return row_indices, column_indices


def _linear_sum_assignment(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if _scipy_linear_sum_assignment is not None:
        return _scipy_linear_sum_assignment(costs)
    return _fallback_linear_sum_assignment(costs)


def match_nodes(
    ground_truth_nodes: list[CellNode] | tuple[CellNode, ...],
    predicted_nodes: list[CellNode] | tuple[CellNode, ...],
    *,
    voxel_spacing: tuple[float, float, float] = DEFAULT_VOXEL_SPACING_UM,
    max_distance_um: float = 7.0,
) -> NodeMatchResult:
    """Optimally match nodes independently by sequence and timepoint.

    Candidate costs are Euclidean distances in physical micrometre space.
    Assignments with distance greater than ``max_distance_um`` are discarded.
    """

    if max_distance_um < 0 or not math.isfinite(max_distance_um):
        raise ValueError("max_distance_um must be finite and non-negative")

    gt_by_key: dict[tuple[str, int], list[CellNode]] = defaultdict(list)
    pred_by_key: dict[tuple[str, int], list[CellNode]] = defaultdict(list)
    for node in ground_truth_nodes:
        gt_by_key[_node_key(node)].append(node)
    for node in predicted_nodes:
        pred_by_key[_node_key(node)].append(node)

    gt_to_pred: dict[str, str] = {}
    pred_to_gt: dict[str, str] = {}
    distances_um: dict[tuple[str, str], float] = {}
    unmatched_gt = {node.node_id for node in ground_truth_nodes}
    unmatched_pred = {node.node_id for node in predicted_nodes}

    for key in sorted(set(gt_by_key) & set(pred_by_key)):
        gt_group = gt_by_key[key]
        pred_group = pred_by_key[key]
        if not gt_group or not pred_group:
            continue

        gt_coords = _physical_array(gt_group, voxel_spacing)
        pred_coords = _physical_array(pred_group, voxel_spacing)
        distances = np.linalg.norm(gt_coords[:, None, :] - pred_coords[None, :, :], axis=2)
        gt_indices, pred_indices = _linear_sum_assignment(distances)
        for gt_index, pred_index in zip(gt_indices, pred_indices, strict=True):
            distance = float(distances[gt_index, pred_index])
            if distance <= max_distance_um:
                gt_id = gt_group[gt_index].node_id
                pred_id = pred_group[pred_index].node_id
                gt_to_pred[gt_id] = pred_id
                pred_to_gt[pred_id] = gt_id
                distances_um[(gt_id, pred_id)] = distance
                unmatched_gt.discard(gt_id)
                unmatched_pred.discard(pred_id)

    return NodeMatchResult(
        gt_to_pred=gt_to_pred,
        pred_to_gt=pred_to_gt,
        distances_um=distances_um,
        unmatched_gt=unmatched_gt,
        unmatched_pred=unmatched_pred,
    )
