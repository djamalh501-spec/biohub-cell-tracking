import pandas as pd

from src.evaluation.validator import regenerate_official_row_ids, validate_submission


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def sample_rows():
    return [
        {"node_id": "a", "sequence_id": "s", "time": 0, "z": 0, "y": 0, "x": 0, "parent_id": ""},
        {"node_id": "b", "sequence_id": "s", "time": 1, "z": 0, "y": 0, "x": 1, "parent_id": "a"},
    ]


def test_valid_submission_passes(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    rows = sample_rows()
    write_csv(sample, rows)
    write_csv(submission, rows)
    result = validate_submission(submission, sample)
    assert result.errors == ()


def test_duplicate_edge_rejection(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    write_csv(sample, sample_rows())
    rows = sample_rows() + [
        {"node_id": "b", "sequence_id": "s", "time": 1, "z": 0, "y": 0, "x": 1, "parent_id": "a"},
    ]
    write_csv(submission, rows)
    result = validate_submission(submission, sample)
    assert any("duplicate edge" in error for error in result.errors)


def test_missing_node_edge_rejection(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    write_csv(sample, sample_rows())
    rows = [
        {"node_id": "b", "sequence_id": "s", "time": 1, "z": 0, "y": 0, "x": 1, "parent_id": "missing"},
    ]
    write_csv(submission, rows)
    result = validate_submission(submission, sample)
    assert any("missing parent" in error for error in result.errors)


def test_backward_time_edge_rejection(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    write_csv(sample, sample_rows())
    rows = [
        {"node_id": "a", "sequence_id": "s", "time": 2, "z": 0, "y": 0, "x": 0, "parent_id": ""},
        {"node_id": "b", "sequence_id": "s", "time": 1, "z": 0, "y": 0, "x": 1, "parent_id": "a"},
    ]
    write_csv(submission, rows)
    result = validate_submission(submission, sample)
    assert any("not forward in time" in error for error in result.errors)


OFFICIAL_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def official_sample_rows():
    return [
        {
            "id": 0,
            "dataset": "44b6_0113de3b",
            "row_type": "node",
            "node_id": 1,
            "t": 0,
            "z": 32,
            "y": 128,
            "x": 128,
            "source_id": -1,
            "target_id": -1,
        },
        {
            "id": 1,
            "dataset": "44b6_0113de3b",
            "row_type": "node",
            "node_id": 2,
            "t": 1,
            "z": 33,
            "y": 128,
            "x": 128,
            "source_id": -1,
            "target_id": -1,
        },
        {
            "id": 2,
            "dataset": "44b6_0113de3b",
            "row_type": "edge",
            "node_id": -1,
            "t": -1,
            "z": -1,
            "y": -1,
            "x": -1,
            "source_id": 1,
            "target_id": 2,
        },
    ]


def write_official_csv(path, rows):
    pd.DataFrame(rows, columns=OFFICIAL_COLUMNS).to_csv(path, index=False)


def test_official_mixed_node_edge_csv_passes(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    rows = official_sample_rows()
    write_official_csv(sample, rows)
    write_official_csv(submission, rows)
    result = validate_submission(submission, sample)
    assert result.errors == ()


def test_duplicate_node_id_allowed_across_different_datasets(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    rows = official_sample_rows()[:1] + [
        {
            "id": 1,
            "dataset": "44b6_0b24845f",
            "row_type": "node",
            "node_id": 1,
            "t": 0,
            "z": 32,
            "y": 128,
            "x": 128,
            "source_id": -1,
            "target_id": -1,
        }
    ]
    write_official_csv(sample, rows)
    write_official_csv(submission, rows)
    result = validate_submission(submission, sample)
    assert result.errors == ()


def test_duplicate_node_id_rejected_within_same_dataset(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    rows = official_sample_rows()[:1] + [
        {
            "id": 1,
            "dataset": "44b6_0113de3b",
            "row_type": "node",
            "node_id": 1,
            "t": 1,
            "z": 33,
            "y": 128,
            "x": 128,
            "source_id": -1,
            "target_id": -1,
        }
    ]
    write_official_csv(sample, official_sample_rows())
    write_official_csv(submission, rows)
    result = validate_submission(submission, sample)
    assert any("duplicate node id" in error for error in result.errors)


def test_edge_referencing_node_from_another_dataset_rejected(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    rows = [
        official_sample_rows()[0],
        {
            "id": 1,
            "dataset": "44b6_0b24845f",
            "row_type": "node",
            "node_id": 2,
            "t": 1,
            "z": 33,
            "y": 128,
            "x": 128,
            "source_id": -1,
            "target_id": -1,
        },
        {
            "id": 2,
            "dataset": "44b6_0113de3b",
            "row_type": "edge",
            "node_id": -1,
            "t": -1,
            "z": -1,
            "y": -1,
            "x": -1,
            "source_id": 1,
            "target_id": 2,
        },
    ]
    write_official_csv(sample, official_sample_rows())
    write_official_csv(submission, rows)
    result = validate_submission(submission, sample)
    assert any("missing child node" in error for error in result.errors)


def test_official_id_column_regenerated_correctly() -> None:
    data = pd.DataFrame(official_sample_rows()).assign(id=[10, 20, 30])
    regenerated = regenerate_official_row_ids(data)
    assert regenerated["id"].tolist() == [0, 1, 2]


def test_official_validator_rejects_nonsequential_ids(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    rows = official_sample_rows()
    rows[1] = {**rows[1], "id": 10}
    write_official_csv(sample, official_sample_rows())
    write_official_csv(submission, rows)
    result = validate_submission(submission, sample)
    assert any("sequential row ids" in error for error in result.errors)


def test_node_rows_with_source_or_target_not_minus_one_rejected(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    rows = official_sample_rows()
    rows[0] = {**rows[0], "source_id": 99}
    rows[1] = {**rows[1], "target_id": 99}
    write_official_csv(sample, official_sample_rows())
    write_official_csv(submission, rows)
    result = validate_submission(submission, sample)
    assert any("node row source_id must be -1" in error for error in result.errors)
    assert any("node row target_id must be -1" in error for error in result.errors)


def test_edge_rows_with_non_minus_one_coordinate_fields_rejected(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    rows = official_sample_rows()
    rows[2] = {**rows[2], "z": 0}
    write_official_csv(sample, official_sample_rows())
    write_official_csv(submission, rows)
    result = validate_submission(submission, sample)
    assert any("edge row t, z, y, x must all be -1" in error for error in result.errors)
