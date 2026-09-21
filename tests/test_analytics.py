import pandas as pd

from src.baseline import train_baseline
from src.evaluation import demographic_parity_difference, subgroup_metrics
from src.data_quality import validate_student_data
from src.report import generate_report
from src.generate_data import generate_student_data


def test_generation_is_reproducible_and_well_formed():
    first = generate_student_data(25, seed=7)
    second = generate_student_data(25, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (25, 10)
    assert first["student_id"].is_unique


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
