import pandas as pd
import pytest

from src.baseline import train_baseline
from src.data_quality import assert_valid_student_data, validate_student_data
from src.evaluation import demographic_parity_difference, subgroup_metrics
from src.generate_data import generate_student_data
from src.report import generate_report


def test_generation_is_reproducible_and_well_formed():
    first = generate_student_data(25, seed=7)
    second = generate_student_data(25, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (25, 10)
    assert list(first.columns) == [
        "student_id",
        "age_band",
        "first_generation",
        "financial_aid",
        "online_student",
        "credits_attempted",
        "attendance_rate",
        "prior_gpa",
        "advising_contacts",
        "persisted_next_term",
    ]
    assert first["student_id"].is_unique
    assert set(first["age_band"]).issubset({"18-20", "21-24", "25+"})
    assert set(first["first_generation"]).issubset({0, 1})
    assert set(first["persisted_next_term"]).issubset({0, 1})


def test_baseline_metrics_and_fairness_table():
    _, results, metrics = train_baseline(generate_student_data(300))
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["brier_score"] <= 1
    table = subgroup_metrics(results, "first_generation", "persisted_next_term", "prediction")
    assert set(table["group"]) == {"0", "1"}
    assert 0 <= demographic_parity_difference(table) <= 1


def test_quality_checks_and_report(tmp_path):
    data = generate_student_data(100)
    assert validate_student_data(data).valid
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)
    report = generate_report(path, tmp_path / "report")
    assert report.exists()
    assert '"false_positive_rate_difference"' in report.read_text()


def test_invalid_schema_is_rejected():
    data = generate_student_data(25, seed=7).drop(columns=["student_id"])
    report = validate_student_data(data)
    assert not report.valid
    assert any("missing columns: student_id" in error for error in report.errors)
    with pytest.raises(ValueError, match="student_id"):
        assert_valid_student_data(data)


def test_invalid_binary_and_range_constraints_are_flagged():
    data = generate_student_data(50, seed=7)
    data.loc[0, "first_generation"] = 2
    data.loc[1, "attendance_rate"] = 150
    report = validate_student_data(data)
    assert not report.valid
    assert any("first_generation must contain only 0/1 values" in error for error in report.errors)
    assert any("attendance_rate contains values outside [0, 100]" in error for error in report.errors)
    with pytest.raises(ValueError):
        assert_valid_student_data(data)


def test_low_row_count_and_missing_values_are_rejected():
    data = generate_student_data(10, seed=7)
    data.loc[0, "prior_gpa"] = None
    report = validate_student_data(data, minimum_rows=20)
    assert not report.valid
    assert any("expected at least 20 rows" in error for error in report.errors)
    assert any("required columns contain missing values" in error for error in report.errors)


def test_duplicate_or_null_student_ids_are_rejected():
    data = generate_student_data(30, seed=7)
    data.loc[0, "student_id"] = data.loc[1, "student_id"]
    report = validate_student_data(data)
    assert not report.valid
    assert any("student_id must be non-null and unique" in error for error in report.errors)
    data.loc[0, "student_id"] = None
    report = validate_student_data(data)
    assert not report.valid
    assert any("student_id must be non-null and unique" in error for error in report.errors)
