from src.evaluation.matching import match_nodes
from src.evaluation.metrics import division_jaccard
from src.evaluation.schema import CellNode, SequenceGraph, TemporalEdge


def n(node_id: str, t: int, x: float = 0.0) -> CellNode:
    return CellNode(node_id=node_id, sequence_id="seq", time=t, z=0.0, y=0.0, x=x)


def graph(nodes: list[CellNode], edges: list[tuple[str, str]]) -> SequenceGraph:
    return SequenceGraph("seq", tuple(nodes), tuple(TemporalEdge(parent, child) for parent, child in edges))


def test_one_correct_division() -> None:
    gt = graph([n("p", 0), n("c1", 1, 0), n("c2", 1, 10)], [("p", "c1"), ("p", "c2")])
    pred = graph([n("pp", 0), n("pc1", 1, 0), n("pc2", 1, 10)], [("pp", "pc1"), ("pp", "pc2")])
    matches = match_nodes(gt.nodes, pred.nodes)
    result = division_jaccard(gt, pred, matches)
    assert result.score == 1.0
    assert result.matched_divisions == 1


def test_missed_division() -> None:
    gt = graph([n("p", 0), n("c1", 1, 0), n("c2", 1, 10)], [("p", "c1"), ("p", "c2")])
    pred = graph([n("pp", 0), n("pc1", 1, 0), n("pc2", 1, 10)], [("pp", "pc1")])
    matches = match_nodes(gt.nodes, pred.nodes)
    result = division_jaccard(gt, pred, matches)
    assert result.score == 0.0
    assert result.false_negative_divisions == 1


def test_false_division() -> None:
    gt = graph([n("p", 0), n("c1", 1, 0), n("c2", 1, 10)], [("p", "c1")])
    pred = graph([n("pp", 0), n("pc1", 1, 0), n("pc2", 1, 10)], [("pp", "pc1"), ("pp", "pc2")])
    matches = match_nodes(gt.nodes, pred.nodes)
    result = division_jaccard(gt, pred, matches)
    assert result.score == 0.0
    assert result.false_positive_divisions == 1


def test_swapped_child_identities_break_exact_division_match() -> None:
    gt = graph([n("p", 0), n("c1", 1, 0), n("c2", 1, 10)], [("p", "c1"), ("p", "c2")])
    pred = graph([n("pp", 0), n("pc1", 1, 0), n("pc2", 1, 10)], [("pp", "pc1"), ("pp", "outside")])
    pred = SequenceGraph(
        "seq",
        pred.nodes + (n("outside", 1, 30),),
        pred.edges,
    )
    matches = match_nodes(gt.nodes, pred.nodes)
    result = division_jaccard(gt, pred, matches)
    assert result.score == 0.0
    assert result.false_positive_divisions == 1
    assert result.false_negative_divisions == 1
