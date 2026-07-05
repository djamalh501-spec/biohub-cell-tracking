from pathlib import Path

import pytest

from src.evaluation.official_metric import read_estimated_number_of_nodes


ROOT = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
TRAIN = ROOT / "train"


EXPECTED_SELF_EVAL = {
    "44b6_0113de3b": {
        "nodes": 52,
        "edges": 50,
        "edge_tp": 50,
        "edge_fp": 0,
        "edge_fn": 0,
        "division_tp": 0,
        "division_fp": 0,
        "division_fn": 0,
        "t_true": 25755,
        "adj": 1.0997980974568045,
    },
    "44b6_0b24845f": {"nodes": 51, "edges": 49, "edge_tp": 49, "t_true": 32795, "adj": 1.099844488489099},
    "44b6_0c582fdc": {"nodes": 71, "edges": 70, "edge_tp": 70, "t_true": 27958, "adj": 1.0997460476428929},
    "44b6_0db75fae": {"nodes": 157, "edges": 151, "edge_tp": 151, "t_true": 15335, "adj": 1.0989761982393218},
    "44b6_12dfb391": {"nodes": 788, "edges": 773, "edge_tp": 773, "division_tp": 1, "t_true": 58672, "adj": 1.0986569402781565},
}


def test_kaggle_official_runtime_self_eval_regression() -> None:
    if not TRAIN.exists():
        pytest.skip("Kaggle train root is not available")

    pytest.importorskip("tracking_cellmot")
    pytest.importorskip("tracksdata")
    pytest.importorskip("geff")
    from tracking_cellmot.io import open_dataset
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise

    rows = []
    for dataset, expected in EXPECTED_SELF_EVAL.items():
        ds = open_dataset(TRAIN / dataset, normalize=False, require_tracks=True, load_image=False)
        graph = ds.tracks
        assert graph is not None
        assert ds.image is None
        assert tuple(ds.scale) == pytest.approx((1.625, 0.40625, 0.40625))
        assert graph.num_nodes() == expected["nodes"]
        assert graph.num_edges() == expected["edges"]

        t_true = read_estimated_number_of_nodes(TRAIN / f"{dataset}.geff")
        assert t_true == pytest.approx(expected["t_true"])

        er = evaluate(graph, graph, scale=tuple(ds.scale))
        assert er.edge_tp == expected["edge_tp"]
        assert er.edge_fp == expected.get("edge_fp", 0)
        assert er.edge_fn == expected.get("edge_fn", 0)
        assert er.division_tp == expected.get("division_tp", 0)
        assert er.division_fp == expected.get("division_fp", 0)
        assert er.division_fn == expected.get("division_fn", 0)

        row = per_sample_metrics(er, t_true, node_recall(graph, graph))
        assert row["edge_jaccard"] == pytest.approx(1.0)
        assert row["node_recall"] == pytest.approx(1.0)
        assert row["adj_edge_jaccard"] == pytest.approx(expected["adj"], abs=1e-12)
        rows.append(row)

    summary = summarise(rows)
    assert summary["edge_jaccard"] == pytest.approx(1.0)
    assert summary["division_jaccard"] == pytest.approx(1.0)
    assert summary["division_tp"] == 1
    assert summary["division_fp"] == 0
    assert summary["division_fn"] == 0
    assert summary["node_recall"] == pytest.approx(1.0)
    assert summary["adj_edge_jaccard"] == pytest.approx(1.0988762387126818, abs=1e-12)
    assert summary["score"] == pytest.approx(1.198876238712682, abs=1e-12)
