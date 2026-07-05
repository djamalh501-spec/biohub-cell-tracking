from src.evaluation.matching import match_nodes
from src.evaluation.metrics import edge_jaccard, raw_edge_jaccard
from src.evaluation.schema import CellNode, SequenceGraph, TemporalEdge


def n(node_id: str, t: int, x: float = 0.0) -> CellNode:
    return CellNode(node_id=node_id, sequence_id="seq", time=t, z=0.0, y=0.0, x=x)


def graph(nodes: list[CellNode], edges: list[tuple[str, str]]) -> SequenceGraph:
    return SequenceGraph("seq", tuple(nodes), tuple(TemporalEdge(parent, child) for parent, child in edges))


def test_perfect_node_and_edge_prediction_has_edge_jaccard_one() -> None:
    gt = graph([n("a", 0), n("b", 1)], [("a", "b")])
    pred = graph([n("pa", 0), n("pb", 1)], [("pa", "pb")])
    matches = match_nodes(gt.nodes, pred.nodes)
    result = edge_jaccard(gt, pred, matches)
    assert result.score == 1.0
    assert result.raw_score == 1.0
    assert result.node_overprediction_penalty == 1.0
    assert result.penalty_formula_verified is False
    assert result.true_positive_edges == 1


def test_no_predictions_has_zero_edge_jaccard_when_gt_has_edges() -> None:
    gt = graph([n("a", 0), n("b", 1)], [("a", "b")])
    pred = graph([], [])
    matches = match_nodes(gt.nodes, pred.nodes)
    result = edge_jaccard(gt, pred, matches)
    assert result.score == 0.0
    assert result.false_negative_edges == 1


def test_all_predictions_unmatched_count_as_false_positive_edges() -> None:
    gt = graph([n("a", 0), n("b", 1)], [("a", "b")])
    pred = graph([n("pa", 0, x=100), n("pb", 1, x=100)], [("pa", "pb")])
    matches = match_nodes(gt.nodes, pred.nodes)
    result = edge_jaccard(gt, pred, matches)
    assert result.score == 0.0
    assert result.false_positive_edges == 1
    assert result.false_negative_edges == 1


def test_edge_jaccard_penalty_approx_reduces_overpredicted_nodes() -> None:
    gt = graph([n("a", 0), n("b", 1)], [("a", "b")])
    pred = graph([n("pa", 0), n("pb", 1), n("extra", 1, x=100)], [("pa", "pb")])
    matches = match_nodes(gt.nodes, pred.nodes)
    raw = raw_edge_jaccard(gt, pred, matches)
    approx = edge_jaccard(gt, pred, matches)
    assert raw.score == 1.0
    assert approx.raw_score == 1.0
    assert approx.node_overprediction_penalty == 2 / 3
    assert approx.score == 2 / 3
