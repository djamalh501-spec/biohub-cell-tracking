import json

import numpy as np
import pytest

from src.evaluation.geff import extract_divisions, read_geff_graph
from src.evaluation.matching import match_nodes
from src.evaluation.metrics import division_jaccard, raw_edge_jaccard


def make_geff_fixture(path, *, edges=None, props=None, prop_names=None) -> None:
    (path / "nodes" / "props").mkdir(parents=True)
    (path / "edges").mkdir(parents=True)
    np.save(path / "nodes" / "ids.npy", np.asarray([1, 2, 3], dtype=np.int64))
    if props is None:
        props = np.asarray(
            [
                [0, 10, 20, 30],
                [1, 11, 20, 30],
                [1, 12, 21, 31],
            ],
            dtype=np.float64,
        )
    np.save(path / "nodes" / "props.npy", props)
    if prop_names is None:
        prop_names = ["t", "z", "y", "x"]
    (path / "nodes" / "props" / "attrs.json").write_text(
        json.dumps({"property_names": prop_names}),
        encoding="utf-8",
    )
    if edges is None:
        edges = np.asarray([[1, 2], [1, 3]], dtype=np.int64)
    np.save(path / "edges" / "ids.npy", edges)
    np.save(path / "edges" / "props.npy", np.empty((len(edges), 0), dtype=np.float64))


def test_read_geff_graph_from_positional_props_fixture(tmp_path) -> None:
    geff = tmp_path / "demo.geff"
    make_geff_fixture(geff)

    graph = read_geff_graph(geff)

    assert graph.sequence_id == "demo"
    assert [node.node_id for node in graph.nodes] == ["1", "2", "3"]
    assert [(node.time, node.z, node.y, node.x) for node in graph.nodes][0] == (0, 10.0, 20.0, 30.0)
    assert [(edge.parent_id, edge.child_id) for edge in graph.edges] == [("1", "2"), ("1", "3")]


def test_read_geff_graph_from_structured_props_fixture(tmp_path) -> None:
    geff = tmp_path / "structured.geff"
    props = np.asarray(
        [(0, 10.0, 20.0, 30.0), (1, 11.0, 20.0, 30.0), (1, 12.0, 21.0, 31.0)],
        dtype=[("t", "i4"), ("z", "f8"), ("y", "f8"), ("x", "f8")],
    )
    make_geff_fixture(geff, props=props)

    graph = read_geff_graph(geff)

    assert graph.nodes[1].time == 1
    assert graph.nodes[1].z == 11.0


def test_read_geff_graph_from_official_grouped_props_fixture(tmp_path) -> None:
    geff = tmp_path / "official.geff"
    (geff / "nodes" / "props").mkdir(parents=True)
    (geff / "edges").mkdir(parents=True)
    np.save(geff / "nodes" / "ids.npy", np.asarray([1, 2, 3], dtype=np.uint64))
    for field in ("t", "z", "y", "x"):
        (geff / "nodes" / "props" / field).mkdir(parents=True)
    np.save(geff / "nodes" / "props" / "t" / "values.npy", np.asarray([0, 1, 1], dtype=np.int64))
    np.save(geff / "nodes" / "props" / "z" / "values.npy", np.asarray([10, 11, 12], dtype=np.int64))
    np.save(geff / "nodes" / "props" / "y" / "values.npy", np.asarray([20, 20, 21], dtype=np.int64))
    np.save(geff / "nodes" / "props" / "x" / "values.npy", np.asarray([30, 30, 31], dtype=np.int64))
    np.save(geff / "edges" / "ids.npy", np.asarray([[1, 2], [1, 3]], dtype=np.uint64))

    graph = read_geff_graph(geff, dataset="official")

    assert graph.sequence_id == "official"
    assert [node.node_id for node in graph.nodes] == ["1", "2", "3"]
    assert [(node.time, node.z, node.y, node.x) for node in graph.nodes] == [
        (0, 10.0, 20.0, 30.0),
        (1, 11.0, 20.0, 30.0),
        (1, 12.0, 21.0, 31.0),
    ]
    assert [(edge.parent_id, edge.child_id) for edge in graph.edges] == [("1", "2"), ("1", "3")]


def test_extract_divisions_infers_exactly_two_children(tmp_path) -> None:
    geff = tmp_path / "division.geff"
    make_geff_fixture(geff)
    graph = read_geff_graph(geff)

    divisions = extract_divisions(graph)

    assert len(divisions) == 1
    assert divisions[0].parent_id == "1"
    assert divisions[0].child_ids == ("2", "3")


def test_geff_decoder_rejects_missing_edge_reference(tmp_path) -> None:
    geff = tmp_path / "bad.geff"
    make_geff_fixture(geff, edges=np.asarray([[1, 99]], dtype=np.int64))

    with pytest.raises(ValueError, match="missing child node"):
        read_geff_graph(geff)


def test_geff_self_evaluation_is_perfect(tmp_path) -> None:
    geff = tmp_path / "self.geff"
    make_geff_fixture(geff)
    graph = read_geff_graph(geff)

    matches = match_nodes(graph.nodes, graph.nodes)
    edge_result = raw_edge_jaccard(graph, graph, matches)
    division_result = division_jaccard(graph, graph, matches)

    assert len(matches.gt_to_pred) == len(graph.nodes)
    assert edge_result.raw_score == 1.0
    assert division_result.score == 1.0
