"""Data-quality checks for student-success training data."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import argparse
from pathlib import Path

REQUIRED_COLUMNS = {
    "student_id", "age_band", "first_generation", "financial_aid",
    "online_student", "credits_attempted", "attendance_rate", "prior_gpa",
    "advising_contacts", "persisted_next_term",
}


@dataclass(frozen=True)
class QualityReport:
    """Serializable result of validating one input frame."""

    valid: bool
    row_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "row_count": self.row_count,
                "errors": list(self.errors), "warnings": list(self.warnings)}


def validate_student_data(data: pd.DataFrame, *, minimum_rows: int = 20) -> QualityReport:
    """Validate schema, uniqueness, missingness, and domain constraints."""
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        errors.append(f"missing columns: {', '.join(missing)}")
        return QualityReport(False, len(data), tuple(errors), tuple(warnings))
    if len(data) < minimum_rows:
        errors.append(f"expected at least {minimum_rows} rows, got {len(data)}")
    if data["student_id"].isna().any() or not data["student_id"].is_unique:
        errors.append("student_id must be non-null and unique")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        errors.append("required columns contain missing values")
    for column in ("first_generation", "financial_aid", "online_student", "persisted_next_term"):
        values = set(data[column].dropna().unique())
        if not values <= {0, 1}:
            errors.append(f"{column} must contain only 0/1 values")
    bounds = {"credits_attempted": (0, 30), "attendance_rate": (0, 100), "prior_gpa": (0, 4),
              "advising_contacts": (0, None)}
    for column, (low, high) in bounds.items():
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.isna().any() or (numeric < low).any() or (high is not None and (numeric > high).any()):
            errors.append(f"{column} contains values outside [{low}, {high if high is not None else 'unbounded'}]")
    if len(data) and data["persisted_next_term"].mean() in (0, 1):
        warnings.append("target contains only one class; model evaluation is not meaningful")
    return QualityReport(not errors, len(data), tuple(errors), tuple(warnings))


def assert_valid_student_data(data: pd.DataFrame, *, minimum_rows: int = 20) -> QualityReport:
    """Validate data and raise a useful error when it cannot be modeled."""
    report = validate_student_data(data, minimum_rows=minimum_rows)
    if not report.valid:
        raise ValueError("; ".join(report.errors))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    report = validate_student_data(pd.read_csv(args.data))
    print(report.as_dict())
    if not report.valid:
        raise SystemExit(1)
