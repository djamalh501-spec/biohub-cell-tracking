import math
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import src.evaluation.official_bridge as official_bridge
from src.evaluation.official_bridge import (
    sequence_graph_from_records,
    submission_df_to_tracksdata,
    tracksdata_to_submission_df,
)
from src.evaluation.official_metric import (
    adjusted_edge_jaccard,
    evaluate_datasets_official_style,
    evaluate_one_dataset,
    read_estimated_number_of_nodes,
)
from src.evaluation.validator import validate_submission


def make_roundtrip_graphs():
    return {
        "dataset_a": sequence_graph_from_records(
            "dataset_a",
            nodes=[
                (1, 0, 10, 20, 30),
                (2, 1, 11, 20, 30),
                (3, 1, 12, 21, 31),
            ],
            edges=[(1, 2), (1, 3)],
        ),
        "dataset_b": sequence_graph_from_records(
            "dataset_b",
            nodes=[
                (1, 0, 3, 4, 5),
                (2, 1, 4, 4, 5),
            ],
            edges=[(1, 2)],
        ),
    }


def test_submission_round_trip_preserves_dataset_scoped_identity(tmp_path) -> None:
    graphs = make_roundtrip_graphs()
    df = tracksdata_to_submission_df(graphs)

    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    df.to_csv(sample, index=False)
    df.to_csv(submission, index=False)
    validation = validate_submission(submission, sample)
    assert validation.errors == ()

    decoded = submission_df_to_tracksdata(pd.read_csv(submission))

    assert set(decoded) == {"dataset_a", "dataset_b"}
    assert [node.node_id for node in decoded["dataset_a"].nodes] == ["1", "2", "3"]
    assert [node.node_id for node in decoded["dataset_b"].nodes] == ["1", "2"]
    assert [(edge.parent_id, edge.child_id) for edge in decoded["dataset_a"].edges] == [("1", "2"), ("1", "3")]
    assert df["id"].tolist() == list(range(len(df)))
    assert set(df.columns) == {"id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"}


def test_self_evaluation_is_perfect_with_t_true() -> None:
    graph = make_roundtrip_graphs()["dataset_a"]
    result = evaluate_one_dataset(graph, graph, t_true=len(graph.nodes))

    assert result.edge_jaccard == 1.0
    assert result.adjusted_edge_jaccard == 1.0
    assert result.division_jaccard == 1.0
    assert result.final_score == 1.1


def test_self_evaluation_without_t_true_marks_adjusted_unavailable() -> None:
    graph = make_roundtrip_graphs()["dataset_a"]
    result = evaluate_one_dataset(graph, graph, t_true=None)

    assert result.edge_jaccard == 1.0
    assert math.isnan(result.adjusted_edge_jaccard)
    assert math.isnan(result.final_score)


def test_adjusted_edge_jaccard_penalty_math() -> None:
    assert adjusted_edge_jaccard(1.0, t_pred=110, t_true=100) == pytest.approx(0.99)
    assert adjusted_edge_jaccard(1.0, t_pred=90, t_true=100) == pytest.approx(1.01)
    assert adjusted_edge_jaccard(1.0, t_pred=2000, t_true=100) == 0.0


def test_dataset_summary_weighting_and_division_micro_average() -> None:
    graphs = make_roundtrip_graphs()
    summary = evaluate_datasets_official_style(
        [(graphs["dataset_a"], graphs["dataset_a"]), (graphs["dataset_b"], graphs["dataset_b"])],
        t_true_map={"dataset_a": 3.0, "dataset_b": 2.0},
    )

    assert summary.adjusted_edge_jaccard == 1.0
    assert summary.division_jaccard == 1.0
    assert summary.final_score == 1.1


def test_read_estimated_number_of_nodes_from_geff_metadata(tmp_path) -> None:
    geff = tmp_path / "demo.geff"
    geff.mkdir()
    (geff / "zarr.json").write_text(
        '{"attributes": {"geff": {"extra": {"estimated_number_of_nodes": 123.5}}}}',
        encoding="utf-8",
    )

    assert read_estimated_number_of_nodes(geff) == 123.5


def test_read_estimated_number_of_nodes_from_zarr_v2_attrs(tmp_path) -> None:
    geff = tmp_path / "demo_v2.geff"
    geff.mkdir()
    (geff / ".zattrs").write_text(
        '{"geff": {"extra": {"estimated_number_of_nodes": 456}}}',
        encoding="utf-8",
    )

    assert read_estimated_number_of_nodes(geff) == 456.0


def test_read_estimated_number_of_nodes_legacy_extra_warns(tmp_path) -> None:
    geff = tmp_path / "legacy.geff"
    geff.mkdir()
    (geff / ".zattrs").write_text(
        '{"extra": {"estimated_number_of_nodes": 789}}',
        encoding="utf-8",
    )

    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert read_estimated_number_of_nodes(geff) == 789.0


def test_read_estimated_number_of_nodes_missing_warns(tmp_path) -> None:
    geff = tmp_path / "missing.geff"
    geff.mkdir()
    (geff / "zarr.json").write_text('{"attributes": {"geff": {"extra": {}}}}', encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="estimated_number_of_nodes not found"):
        assert read_estimated_number_of_nodes(geff) is None


def test_tracksdata_builder_prefers_indexed_graph_and_ignores_predefined_attrs(monkeypatch) -> None:
    class FakeIndexedRXGraph:
        def __init__(self) -> None:
            self.node_attr_keys = {"t", "z", "y", "x"}
            self.nodes: list[dict[str, object]] = []
            self.indices: list[int] = []
            self.edges: list[dict[str, int]] = []

        def add_node_attr_key(self, key, dtype, default_value=None) -> None:
            if key in self.node_attr_keys:
                raise ValueError(f"Attribute key {key} already exists")
            self.node_attr_keys.add(key)

        def add_edge_attr_key(self, key, dtype, default_value=None) -> None:
            raise AssertionError("edge attr keys are not registered by the bridge yet")

        def bulk_add_nodes(self, nodes, indices=None) -> None:
            self.nodes = list(nodes)
            self.indices = list(indices)

        def bulk_add_edges(self, edges) -> None:
            self.edges = list(edges)

    class FakeInMemoryGraph:
        def __init__(self) -> None:
            raise AssertionError("IndexedRXGraph should be preferred when available")

    fake_tracksdata = SimpleNamespace(
        graph=SimpleNamespace(IndexedRXGraph=FakeIndexedRXGraph, InMemoryGraph=FakeInMemoryGraph)
    )
    fake_polars = SimpleNamespace(Int64=object(), Float64=object())
    monkeypatch.setitem(sys.modules, "polars", fake_polars)
    monkeypatch.setattr(official_bridge, "_tracksdata", fake_tracksdata)

    df = tracksdata_to_submission_df({"dataset_a": make_roundtrip_graphs()["dataset_a"]})
    graphs = submission_df_to_tracksdata(df)

    graph = graphs["dataset_a"]
    assert isinstance(graph, FakeIndexedRXGraph)
    assert graph.indices == [1, 2, 3]
    assert graph.nodes == [
        {"t": 0, "z": 10.0, "y": 20.0, "x": 30.0},
        {"t": 1, "z": 11.0, "y": 20.0, "x": 30.0},
        {"t": 1, "z": 12.0, "y": 21.0, "x": 31.0},
    ]
    assert graph.edges == [{"source_id": 1, "target_id": 2}, {"source_id": 1, "target_id": 3}]


def test_safe_attr_helpers_reraise_unexpected_value_errors() -> None:
    class FakeGraph:
        def add_node_attr_key(self, key, dtype, default_value=None) -> None:
            raise ValueError("different failure")

        def add_edge_attr_key(self, key, dtype, default_value=None) -> None:
            raise ValueError("different failure")

    with pytest.raises(ValueError, match="different failure"):
        official_bridge._safe_add_node_attr_key(FakeGraph(), "t", object(), default_value=0)
    with pytest.raises(ValueError, match="different failure"):
        official_bridge._safe_add_edge_attr_key(FakeGraph(), "source_id", object(), default_value=-1)


def test_tracking_cellmot_cross_check_skips_when_unavailable() -> None:
    pytest.importorskip("tracking_cellmot")
    # The official package is optional and heavy. Once installed, this test should be
    # expanded with the official tracksdata constructor observed in that environment.
    assert True
