import pandas as pd
import numpy as np
from types import SimpleNamespace

import src.baseline.make_baseline_submission as baseline
import src.baseline.sweep_adaptive as sweep
from src.baseline.make_baseline_submission import (
    Detection,
    adaptive_target_count,
    build_parser,
    causal_rolling_median_target,
    detect_frame_peaks,
    detect_frame_peaks_adaptive,
    graph_from_detections,
    graphs_to_submission,
    inspect_zarr_image,
    link_detections_one_to_one,
    main,
    make_classical_graph,
    run_eval_on_train,
)
from src.baseline.sweep_adaptive import parse_official_summary
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


def test_detection_policy_cli_defaults_to_fixed() -> None:
    parser = build_parser()
    args = parser.parse_args(["--mode", "classical", "--test-root", "test"])
    assert args.detection_policy == "fixed"
    assert args.adaptive_mad_k == 3.0
    assert args.adaptive_min_detections == 80
    assert args.adaptive_max_detections == 400
    assert args.adaptive_count_smoothing_window == 5


def test_fixed_policy_regression_uses_reference_detector(tmp_path, monkeypatch) -> None:
    store = tmp_path / "dataset_a.zarr"
    store.mkdir()
    frame = np.zeros((1, 3, 16, 16), dtype=np.float32)
    frame[0, 1, 4, 5] = 10
    frame[0, 1, 10, 11] = 8
    np.save(store / "volume.npy", frame)
    monkeypatch.setattr(
        baseline,
        "detect_frame_peaks_adaptive",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("adaptive detector called in fixed mode")),
    )
    args = SimpleNamespace(
        detection_policy="fixed",
        threshold_percentile=99.0,
        min_distance_xy=2,
        min_distance_z=1,
        max_detections_per_frame=5,
        sigma=0,
        link_max_um=2.0,
    )
    graph = make_classical_graph(inspect_zarr_image(store), args)
    assert [(node.time, node.z, node.y, node.x) for node in graph.nodes] == [
        (0, 1.0, 4.0, 5.0),
        (0, 1.0, 10.0, 11.0),
    ]


def test_adaptive_target_count_respects_bounds() -> None:
    scores = np.arange(1, 1001, dtype=float)
    lower_target, _ = adaptive_target_count(scores, mad_k=100.0, minimum=80, maximum=400)
    upper_target, _ = adaptive_target_count(scores, mad_k=0.0, minimum=80, maximum=400)
    assert lower_target == 80
    assert upper_target == 400


def test_adaptive_target_count_handles_zero_mad_and_nonfinite_scores() -> None:
    target, threshold = adaptive_target_count(
        [1.0, 1.0, 1.0, 1.0, np.nan, np.inf],
        mad_k=3.0,
        minimum=3,
        maximum=10,
    )
    assert threshold == 1.0
    assert target == 3


def test_adaptive_detector_handles_empty_candidate_frame() -> None:
    detections, next_node_id, diagnostics = detect_frame_peaks_adaptive(
        np.zeros((2, 8, 8), dtype=np.float32),
        t=0,
        next_node_id=1,
        threshold_percentile=99.7,
        min_distance_xy=2,
        min_distance_z=1,
        sigma=0,
        adaptive_mad_k=3.0,
        adaptive_min_detections=2,
        adaptive_max_detections=5,
        adaptive_count_smoothing_window=5,
        raw_target_history=[],
    )
    assert detections == []
    assert next_node_id == 1
    assert diagnostics.candidate_count == 0
    assert diagnostics.raw_target == 0
    assert diagnostics.selected_count == 0


def test_adaptive_detector_is_deterministic() -> None:
    frame = np.zeros((3, 16, 16), dtype=np.float32)
    frame[1, 3, 4] = 10
    frame[1, 8, 9] = 8
    frame[1, 13, 13] = 6
    kwargs = {
        "t": 0,
        "next_node_id": 1,
        "threshold_percentile": 90.0,
        "min_distance_xy": 2,
        "min_distance_z": 1,
        "sigma": 0,
        "adaptive_mad_k": 3.0,
        "adaptive_min_detections": 2,
        "adaptive_max_detections": 3,
        "adaptive_count_smoothing_window": 3,
        "raw_target_history": [],
    }
    assert detect_frame_peaks_adaptive(frame, **kwargs) == detect_frame_peaks_adaptive(frame, **kwargs)


def test_causal_count_smoothing_uses_available_history() -> None:
    assert causal_rolling_median_target([10], 5) == 10
    assert causal_rolling_median_target([10, 100], 5) == 55
    assert causal_rolling_median_target([10, 100, 20], 3) == 20
    assert causal_rolling_median_target([10, 100, 20, 30], 3) == 30


def test_adaptive_mode_writes_valid_submission_for_dynamic_datasets(tmp_path) -> None:
    test_root = tmp_path / "test"
    test_root.mkdir()
    for index, dataset in enumerate(("unseen_alpha", "unseen_beta")):
        store = test_root / f"{dataset}.zarr"
        store.mkdir()
        volume = np.zeros((2, 2, 8, 8), dtype=np.float32)
        volume[:, 0, 2, 2] = 10 + index
        volume[:, 1, 6, 6] = 8 + index
        np.save(store / "volume.npy", volume)
    sample = tmp_path / "sample_submission.csv"
    output = tmp_path / "submission.csv"
    _sample_submission(sample, datasets=("unseen_alpha", "unseen_beta"))

    result = main(
        [
            "--mode",
            "classical",
            "--test-root",
            str(test_root),
            "--sample-submission",
            str(sample),
            "--output",
            str(output),
            "--detection-policy",
            "adaptive-count",
            "--threshold-percentile",
            "90",
            "--sigma",
            "0",
            "--min-distance-xy",
            "1",
            "--min-distance-z",
            "0",
            "--adaptive-min-detections",
            "1",
            "--adaptive-max-detections",
            "2",
            "--adaptive-count-smoothing-window",
            "1",
        ]
    )
    assert result == 0
    assert validate_submission(output, sample).errors == ()
    submission = pd.read_csv(output)
    assert tuple(submission.columns) == OFFICIAL_SUBMISSION_COLUMNS
    assert set(submission["dataset"]) == {"unseen_alpha", "unseen_beta"}


def test_sweep_summary_parser() -> None:
    output = (
        "OFFICIAL sweep_metrics: dataset_count=3 score=0.18003 "
        "node_recall=0.25000 adj_edge_jaccard=0.18003\n"
    )
    assert parse_official_summary(output) == {
        "dataset_count": 3,
        "official_summary_score": 0.18003,
        "node_recall": 0.25,
        "adjusted_edge_jaccard": 0.18003,
    }


def test_sweep_propagates_parameters_and_writes_success_csv(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "OFFICIAL sweep_metrics: dataset_count=3 score=0.18123456789 "
                "node_recall=0.25 adj_edge_jaccard=0.18123456789\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    output = tmp_path / "sweep.csv"
    result = sweep.main(
        [
            "--train-root",
            str(tmp_path / "train"),
            "--output",
            str(output),
            "--limit",
            "3",
            "--config",
            "2.75,350,6.0",
            "--threshold-percentile",
            "99.6",
            "--sigma",
            "1.25",
            "--min-distance-xy",
            "6",
            "--min-distance-z",
            "3",
            "--adaptive-min-detections",
            "70",
            "--adaptive-count-smoothing-window",
            "7",
            "--seed",
            "123",
        ]
    )
    assert result == 0
    command = captured["command"]
    assert isinstance(command, list)
    for option, value in (
        ("--detection-policy", "adaptive-count"),
        ("--threshold-percentile", "99.6"),
        ("--sigma", "1.25"),
        ("--min-distance-xy", "6"),
        ("--min-distance-z", "3"),
        ("--adaptive-mad-k", "2.75"),
        ("--adaptive-min-detections", "70"),
        ("--adaptive-max-detections", "350"),
        ("--adaptive-count-smoothing-window", "7"),
        ("--link-max-um", "6.0"),
        ("--seed", "123"),
    ):
        index = command.index(option)
        assert command[index + 1] == value
    assert captured["kwargs"] == {"capture_output": True, "text": True, "check": False}
    row = pd.read_csv(output).iloc[0]
    assert row["status"] == "success"
    assert row["dataset_count"] == 3
    assert row["official_summary_score"] == 0.18123456789


def test_sweep_writes_failure_csv_when_subprocess_cannot_launch(tmp_path, monkeypatch) -> None:
    def fail_to_launch(*args, **kwargs):
        raise OSError("runtime unavailable")

    monkeypatch.setattr(sweep.subprocess, "run", fail_to_launch)
    output = tmp_path / "failed_sweep.csv"
    result = sweep.main(
        [
            "--train-root",
            str(tmp_path / "train"),
            "--output",
            str(output),
            "--config",
            "3.0,400,5.0",
        ]
    )
    assert result == 1
    row = pd.read_csv(output).iloc[0]
    assert row["status"] == "failure"
    assert row["return_code"] == -1
    assert "could not launch eval-on-train subprocess" in row["error"]


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
