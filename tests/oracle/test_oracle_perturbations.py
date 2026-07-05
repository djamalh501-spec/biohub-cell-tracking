import pandas as pd
import pytest

from src.evaluation.official_bridge import sequence_graph_from_records, tracksdata_to_submission_df
from src.evaluation.validator import validate_submission
from src.oracle.oracle_experiments import (
    coord_jitter,
    division_removal,
    edge_swap,
    node_count_adjustment_curve,
    node_dropout,
)


def make_submission_df() -> pd.DataFrame:
    graph = sequence_graph_from_records(
        "dataset_a",
        nodes=[
            (1, 0, 10, 20, 30),
            (2, 1, 11, 20, 30),
            (3, 1, 12, 21, 31),
            (4, 1, 13, 22, 32),
            (5, 2, 14, 22, 32),
        ],
        edges=[(1, 2), (1, 3), (2, 5)],
    )
    return tracksdata_to_submission_df({"dataset_a": graph})


def assert_valid(tmp_path, df: pd.DataFrame) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    make_submission_df().to_csv(sample, index=False)
    df.to_csv(submission, index=False)
    result = validate_submission(submission, sample)
    assert result.errors == ()


def test_node_dropout_removes_incident_edges(tmp_path, rng) -> None:
    df = make_submission_df()
    dropped = node_dropout(df, 0.5, rng)

    assert set(dropped[dropped["row_type"] == "node"]["node_id"]) == {1, 3, 4}
    assert set(map(tuple, dropped[dropped["row_type"] == "edge"][["source_id", "target_id"]].to_numpy())) == {(1, 3)}
    assert dropped["id"].tolist() == list(range(len(dropped)))
    assert_valid(tmp_path, dropped)


def test_coord_jitter_changes_coordinates_but_preserves_schema(tmp_path, rng) -> None:
    df = make_submission_df()
    jittered = coord_jitter(df, 10.0, rng, image_shape=(3, 20, 64, 64))

    assert list(jittered.columns) == list(df.columns)
    assert not jittered.loc[jittered["row_type"] == "node", ["z", "y", "x"]].equals(
        df.loc[df["row_type"] == "node", ["z", "y", "x"]]
    )
    assert jittered.loc[jittered["row_type"] == "edge", ["node_id", "t", "z", "y", "x"]].eq(-1).all().all()
    assert_valid(tmp_path, jittered)


def test_node_count_adjustment_curve_formula() -> None:
    curve = node_count_adjustment_curve(100, ratios=[0.001, 1.0, 20.0])

    assert curve.loc[0, "adj_edge_jaccard"] == pytest.approx(1.0999)
    assert curve.loc[1, "adj_edge_jaccard"] == pytest.approx(1.0)
    assert curve.loc[2, "adj_edge_jaccard"] == 0.0


def test_edge_swap_preserves_forward_time_validity(tmp_path, rng) -> None:
    df = make_submission_df()
    swapped = edge_swap(df, 1.0, rng)

    edges = swapped[swapped["row_type"] == "edge"]
    nodes = swapped[swapped["row_type"] == "node"].set_index("node_id")
    for edge in edges.itertuples(index=False):
        assert int(nodes.loc[edge.source_id, "t"]) < int(nodes.loc[edge.target_id, "t"])
    assert set(map(tuple, edges[["source_id", "target_id"]].to_numpy())) != {(1, 2), (1, 3), (2, 5)}
    assert_valid(tmp_path, swapped)


def test_division_removal_removes_one_daughter_edge(tmp_path) -> None:
    df = make_submission_df()
    removed = division_removal(df)

    source_one_edges = removed[(removed["row_type"] == "edge") & (removed["source_id"] == 1)]
    assert len(source_one_edges) == 1
    assert_valid(tmp_path, removed)
