"""Typed lineage graph schema and validation helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Mapping


DEFAULT_VOXEL_SPACING_UM: tuple[float, float, float] = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True, slots=True)
class CellNode:
    """A tracked cell nucleus at one timepoint in voxel coordinates.

    Coordinates are explicitly stored in ``(z, y, x)`` order.
    """

    node_id: str
    sequence_id: str
    time: int
    z: float
    y: float
    x: float


@dataclass(frozen=True, slots=True)
class TemporalEdge:
    """A directed lineage edge from an earlier parent node to a later child node."""

    parent_id: str
    child_id: str


@dataclass(frozen=True, slots=True)
class LineageDivision:
    """A parent with two or more child nodes."""

    parent_id: str
    child_ids: tuple[str, ...]


@dataclass(slots=True)
class SequenceGraph:
    """Lineage graph for one independent embryo or sequence."""

    sequence_id: str
    nodes: tuple[CellNode, ...]
    edges: tuple[TemporalEdge, ...]

    @property
    def node_by_id(self) -> dict[str, CellNode]:
        return {node.node_id: node for node in self.nodes}

    @property
    def divisions(self) -> tuple[LineageDivision, ...]:
        children_by_parent: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            children_by_parent[edge.parent_id].append(edge.child_id)
        return tuple(
            LineageDivision(parent_id, tuple(sorted(child_ids)))
            for parent_id, child_ids in sorted(children_by_parent.items())
            if len(child_ids) >= 2
        )

    def validate(self, *, allow_multiple_parents: bool = False) -> list[str]:
        """Return validation errors for this graph."""

        errors: list[str] = []
        ids = [node.node_id for node in self.nodes]
        duplicate_ids = sorted(node_id for node_id, count in Counter(ids).items() if count > 1)
        for node_id in duplicate_ids:
            errors.append(f"duplicate node id within sequence {self.sequence_id!r}: {node_id!r}")

        nodes = self.node_by_id
        for node in self.nodes:
            if node.sequence_id != self.sequence_id:
                errors.append(
                    f"node {node.node_id!r} has sequence_id {node.sequence_id!r}, "
                    f"expected {self.sequence_id!r}"
                )
            if not isinstance(node.time, int):
                errors.append(f"node {node.node_id!r} has non-integer time {node.time!r}")
            if not all(math.isfinite(value) for value in (node.z, node.y, node.x)):
                errors.append(f"node {node.node_id!r} has non-finite coordinates")

        edge_pairs = [(edge.parent_id, edge.child_id) for edge in self.edges]
        duplicate_edges = sorted(pair for pair, count in Counter(edge_pairs).items() if count > 1)
        for parent_id, child_id in duplicate_edges:
            errors.append(f"duplicate edge {parent_id!r}->{child_id!r}")

        parent_by_child: dict[str, list[str]] = defaultdict(list)
        children_by_parent: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            if edge.parent_id == edge.child_id:
                errors.append(f"self edge is not allowed for node {edge.parent_id!r}")
            parent = nodes.get(edge.parent_id)
            child = nodes.get(edge.child_id)
            if parent is None:
                errors.append(f"edge references missing parent node {edge.parent_id!r}")
            if child is None:
                errors.append(f"edge references missing child node {edge.child_id!r}")
            if parent is None or child is None:
                continue
            if parent.time >= child.time:
                errors.append(
                    f"edge {edge.parent_id!r}->{edge.child_id!r} is not forward in time "
                    f"({parent.time} >= {child.time})"
                )
            parent_by_child[edge.child_id].append(edge.parent_id)
            children_by_parent[edge.parent_id].append(edge.child_id)

        if not allow_multiple_parents:
            for child_id, parent_ids in sorted(parent_by_child.items()):
                if len(set(parent_ids)) > 1:
                    errors.append(f"child node {child_id!r} has multiple parents: {sorted(set(parent_ids))}")

        for parent_id, child_ids in sorted(children_by_parent.items()):
            unique_children = set(child_ids)
            if len(unique_children) > 2:
                errors.append(
                    f"parent node {parent_id!r} has {len(unique_children)} children; "
                    "division out-degree greater than two is unsupported"
                )

        return errors


def group_nodes_by_sequence(nodes: Iterable[CellNode]) -> dict[str, list[CellNode]]:
    grouped: dict[str, list[CellNode]] = defaultdict(list)
    for node in nodes:
        grouped[node.sequence_id].append(node)
    return dict(grouped)


def build_graphs_by_sequence(
    nodes: Iterable[CellNode],
    edges: Iterable[TemporalEdge],
    *,
    edge_sequence_ids: Mapping[tuple[str, str], str] | None = None,
) -> dict[str, SequenceGraph]:
    """Build sequence graphs from flat nodes and edges.

    If ``edge_sequence_ids`` is omitted, each edge is assigned from its parent node.
    """

    grouped_nodes = group_nodes_by_sequence(nodes)
    node_sequence = {node.node_id: node.sequence_id for node in nodes}
    grouped_edges: dict[str, list[TemporalEdge]] = defaultdict(list)
    for edge in edges:
        sequence_id = None
        if edge_sequence_ids is not None:
            sequence_id = edge_sequence_ids.get((edge.parent_id, edge.child_id))
        if sequence_id is None:
            sequence_id = node_sequence.get(edge.parent_id, "")
        grouped_edges[sequence_id].append(edge)
    sequence_ids = sorted(set(grouped_nodes) | set(grouped_edges))
    return {
        sequence_id: SequenceGraph(
            sequence_id=sequence_id,
            nodes=tuple(grouped_nodes.get(sequence_id, ())),
            edges=tuple(grouped_edges.get(sequence_id, ())),
        )
        for sequence_id in sequence_ids
    }
