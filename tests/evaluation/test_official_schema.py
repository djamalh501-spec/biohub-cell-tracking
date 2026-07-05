import pandas as pd

from src.evaluation.official_schema import OFFICIAL_SUBMISSION_COLUMNS, read_sample_submission_schema, read_submission_csv


def test_read_sample_submission_schema_records_exact_columns(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        [{"node_id": "a", "sequence_id": "s", "time": 0, "z": 0, "y": 0, "x": 0, "parent_id": ""}]
    ).to_csv(sample, index=False)

    schema = read_sample_submission_schema(sample)

    assert schema.columns == ("node_id", "sequence_id", "time", "z", "y", "x", "parent_id")
    assert schema.row_count == 1
    assert schema.exact_official_columns_verified is False


def test_read_sample_submission_schema_verifies_official_columns(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    rows = [
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
        }
    ]
    pd.DataFrame(rows, columns=OFFICIAL_SUBMISSION_COLUMNS).to_csv(sample, index=False)

    schema = read_sample_submission_schema(sample)

    assert schema.columns == OFFICIAL_SUBMISSION_COLUMNS
    assert schema.exact_official_columns_verified is True


def test_read_submission_csv_converts_rows_to_graphs(tmp_path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    rows = [
        {"node_id": "a", "sequence_id": "s", "time": 0, "z": 0, "y": 0, "x": 0, "parent_id": ""},
        {"node_id": "b", "sequence_id": "s", "time": 1, "z": 0, "y": 0, "x": 1, "parent_id": "a"},
    ]
    pd.DataFrame(rows).to_csv(sample, index=False)
    pd.DataFrame(rows).to_csv(submission, index=False)

    _, graphs, schema, warnings = read_submission_csv(submission, sample)

    assert schema.columns == ("node_id", "sequence_id", "time", "z", "y", "x", "parent_id")
    assert warnings == ()
    assert len(graphs["s"].nodes) == 2
    assert len(graphs["s"].edges) == 1
