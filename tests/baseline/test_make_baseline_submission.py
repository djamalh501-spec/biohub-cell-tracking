import pandas as pd
import numpy as np

import src.baseline.make_baseline_submission as baseline
from src.baseline.make_baseline_submission import (
    Detection,
    detect_frame_peaks,
    graph_from_detections,
    graphs_to_submission,
    link_detections_one_to_one,
    main,
    run_eval_on_train,
)
from src.evaluation.official_bridge import sequence_graph_from_records
from src.evaluation.validator import OFFICIAL_SUBMISSION_COLUMNS, validate_submission


def _sample_submission(path, datasets=("dataset_a",)):
    rows = []
    row_id = 0
    for dataset in datasets:
        rows.append(
            {
                "id": row_id,
                "dataset": dataset,
                "row_type": "node",
                "node_id": 1,
                "t": 0,
                "z": 0,
                "y": 0,
                "x": 0,
                "source_id": -1,
                "target_id": -1,
            }
        )
        row_id += 1
    pd.DataFrame(rows, columns=OFFICIAL_SUBMISSION_COLUMNS).to_csv(path, index=False)


def _simple_submission_frame():
    graph = graph_from_detections(
        "dataset_a",
        [
            Detection(1, 0, 1, 2, 3, 1.0),
            Detection(2, 1, 1, 3, 4, 1.0),
            Detection(3, 0, 1, 5, 6, 1.0),
        ],
        [(1, 2)],
    )
    return graphs_to_submission([graph])


def test_output_columns_exactly_match_official_schema() -> None:
    df = _simple_submission_frame()
    assert tuple(df.columns) == OFFICIAL_SUBMISSION_COLUMNS


def test_ids_are_sequential() -> None:
    df = _simple_submission_frame()
    assert df["id"].tolist() == list(range(len(df)))


def test_node_id_uniqueness_per_dataset() -> None:
    df = _simple_submission_frame()
    node_rows = df[df["row_type"] == "node"]
    assert not node_rows.duplicated(["dataset", "node_id"]).any()


def test_edges_reference_existing_nodes_in_same_dataset() -> None:
    df = _simple_submission_frame()
    node_ids = set(df.loc[df["row_type"] == "node", "node_id"])
    edge_rows = df[df["row_type"] == "edge"]
    assert set(edge_rows["source_id"]).issubset(node_ids)
    assert set(edge_rows["target_id"]).issubset(node_ids)


def test_edges_go_forward_in_time() -> None:
    df = _simple_submission_frame()
    node_time = {int(row.node_id): int(row.t) for row in df[df["row_type"] == "node"].itertuples(index=False)}
    for edge in df[df["row_type"] == "edge"].itertuples(index=False):
        assert node_time[int(edge.source_id)] < node_time[int(edge.target_id)]


def test_node_rows_use_sentinel_values() -> None:
    nodes = _simple_submission_frame().query("row_type == 'node'")
    assert (nodes["source_id"] == -1).all()
    assert (nodes["target_id"] == -1).all()


def test_edge_rows_use_sentinel_values() -> None:
    edges = _simple_submission_frame().query("row_type == 'edge'")
    assert (edges[["node_id", "t", "z", "y", "x"]] == -1).all().all()


def test_output_passes_existing_validator(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    output = tmp_path / "submission.csv"
    _sample_submission(sample)
    _simple_submission_frame().to_csv(output, index=False)
    result = validate_submission(output, sample)
    assert result.errors == ()


def test_smoke_mode_works_on_synthetic_zarr_fixtures(tmp_path) -> None:
    test_root = tmp_path / "test"
    test_root.mkdir()
    for dataset in ("dataset_a", "dataset_b"):
        store = test_root / f"{dataset}.zarr"
        store.mkdir()
        np.save(store / "volume.npy", np.zeros((2, 3, 8, 8), dtype=np.uint16))
    sample = tmp_path / "sample_submission.csv"
    output = tmp_path / "submission.csv"
    _sample_submission(sample, datasets=("dataset_a", "dataset_b"))

    assert main(["--mode", "smoke", "--test-root", str(test_root), "--sample-submission", str(sample), "--output", str(output)]) == 0
    result = validate_submission(output, sample)
    assert result.errors == ()
    df = pd.read_csv(output)
    assert set(df["dataset"]) == {"dataset_a", "dataset_b"}


def test_classical_detector_is_deterministic_on_known_blobs() -> None:
    frame = np.zeros((3, 16, 16), dtype=np.float32)
    frame[1, 4, 5] = 10
    frame[1, 10, 11] = 8

    first, _ = detect_frame_peaks(
        frame,
        t=0,
        next_node_id=1,
        threshold_percentile=99.0,
        min_distance_xy=2,
        min_distance_z=1,
        max_detections=5,
        sigma=0,
    )
    second, _ = detect_frame_peaks(
        frame,
        t=0,
        next_node_id=1,
        threshold_percentile=99.0,
        min_distance_xy=2,
        min_distance_z=1,
        max_detections=5,
        sigma=0,
    )

    assert first == second
    assert [(det.z, det.y, det.x) for det in first] == [(1, 4, 5), (1, 10, 11)]


def test_linker_one_to_one_assignment_recovers_known_correspondences() -> None:
    detections_by_t = {
        0: [Detection(1, 0, 0, 0, 0, 1), Detection(2, 0, 0, 10, 0, 1)],
        1: [Detection(3, 1, 0, 1, 0, 1), Detection(4, 1, 0, 11, 0, 1)],
    }
    edges = link_detections_one_to_one(detections_by_t, scale_zyx_um=(1.0, 1.0, 1.0), link_max_um=2.0)
    assert edges == [(1, 3), (2, 4)]


def test_linker_has_no_duplicate_targets_in_one_step() -> None:
    detections_by_t = {
        0: [Detection(1, 0, 0, 0, 0, 1), Detection(2, 0, 0, 1, 0, 1)],
        1: [Detection(3, 1, 0, 0, 0, 1)],
    }
    edges = link_detections_one_to_one(detections_by_t, scale_zyx_um=(1.0, 1.0, 1.0), link_max_um=5.0)
    targets = [target for _, target in edges]
    assert len(targets) == len(set(targets))


def test_eval_on_train_prefers_official_scoring_when_available(tmp_path, monkeypatch, capsys) -> None:
    train_root = tmp_path / "train"
    train_root.mkdir()
    zarr_path = train_root / "dataset_a.zarr"
    zarr_path.mkdir()
    np.save(zarr_path / "volume.npy", np.zeros((1, 2, 4, 4), dtype=np.uint16))
    (train_root / "dataset_a.geff").mkdir()
    pred_graph = sequence_graph_from_records("dataset_a", [(1, 0, 0, 0, 0)], [])
    gt_graph = object()
    pred_tracks_graph = object()
    calls = {"bridge": 0, "open_dataset": 0, "evaluate": 0, "fallback_geff": 0}

    monkeypatch.setattr(baseline, "make_classical_graph", lambda info, args: pred_graph)
    monkeypatch.setattr(baseline, "read_estimated_number_of_nodes", lambda path: 10.0)

    def fake_submission_df_to_tracksdata(df):
        calls["bridge"] += 1
        assert tuple(df.columns) == OFFICIAL_SUBMISSION_COLUMNS
        return {"dataset_a": pred_tracks_graph}

    def fake_open_dataset(path, normalize, require_tracks, load_image):
        calls["open_dataset"] += 1
        assert path == train_root / "dataset_a"
        assert normalize is False
        assert require_tracks is True
        assert load_image is False
        return type("Dataset", (), {"tracks": gt_graph, "scale": (1.0, 1.0, 1.0)})()

    def fake_evaluate(pred, gt, scale):
        calls["evaluate"] += 1
        assert pred is pred_tracks_graph
        assert gt is gt_graph
        assert scale == (1.0, 1.0, 1.0)
        return object()

    monkeypatch.setattr(baseline, "submission_df_to_tracksdata", fake_submission_df_to_tracksdata)
    monkeypatch.setattr(baseline, "open_dataset", fake_open_dataset)
    monkeypatch.setattr(baseline, "read_geff_graph", lambda *args, **kwargs: calls.__setitem__("fallback_geff", 1))
    monkeypatch.setattr(baseline.official_metric_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(baseline.official_metric_module, "node_recall", lambda pred, gt: 0.5)
    monkeypatch.setattr(
        baseline.official_metric_module,
        "per_sample_metrics",
        lambda er, t_true, recall: {
            "edge_jaccard": 0.25,
            "adj_edge_jaccard": 0.3,
            "node_recall": recall,
            "division_tp": 0,
            "division_fp": 0,
            "division_fn": 0,
        },
    )
    monkeypatch.setattr(baseline.official_metric_module, "summarise", lambda rows: {"score": 0.3, "n": len(rows)})

    args = type(
        "Args",
        (),
        {
            "train_root": train_root,
            "limit": 1,
            "threshold_percentile": 99.5,
            "min_distance_xy": 5,
            "min_distance_z": 2,
            "max_detections_per_frame": 10,
            "sigma": 0,
            "link_max_um": 5.0,
        },
    )()

    assert run_eval_on_train(args) == 0
    output = capsys.readouterr().out
    assert "OFFICIAL dataset_a" in output
    assert "OFFICIAL summary" in output
    assert calls == {"bridge": 1, "open_dataset": 1, "evaluate": 1, "fallback_geff": 0}
