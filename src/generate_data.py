"""Generate deterministic synthetic student success data."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_student_data(n_students: int = 500, seed: int = 42) -> pd.DataFrame:
    """Return synthetic, non-identifying student records."""
    rng = np.random.default_rng(seed)
    first_generation = rng.binomial(1, 0.38, n_students)
    financial_aid = rng.binomial(1, 0.52, n_students)
    online = rng.binomial(1, 0.35, n_students)
    credits = np.clip(rng.normal(13, 3, n_students).round(), 3, 21)
    attendance = np.clip(rng.normal(82 - 5 * online, 10, n_students), 45, 100)
    prior_gpa = np.clip(rng.normal(3.0, 0.55, n_students), 1.2, 4.0)
    advising_contacts = rng.poisson(2.2, n_students)
    age_band = rng.choice(["18-20", "21-24", "25+"], n_students, p=[0.58, 0.27, 0.15])

    risk_score = (
        -0.8
        + 0.9 * (prior_gpa - 2.5)
        + 0.035 * (attendance - 75)
        + 0.08 * advising_contacts
        + 0.06 * (credits - 12)
        - 0.35 * first_generation
        - 0.25 * financial_aid
        - 0.3 * online
    )
    probability = 1 / (1 + np.exp(-risk_score))
    persisted = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "student_id": [f"S{i:04d}" for i in range(1, n_students + 1)],
            "age_band": age_band,
            "first_generation": first_generation,
            "financial_aid": financial_aid,
            "online_student": online,
            "credits_attempted": credits.astype(int),
            "attendance_rate": attendance.round(1),
            "prior_gpa": prior_gpa.round(2),
            "advising_contacts": advising_contacts,
            "persisted_next_term": persisted,
        }
    )


def write_synthetic_data(path: str | Path, n_students: int = 500, seed: int = 42) -> Path:
    """Generate and write a CSV, creating its parent directory if needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_student_data(n_students=n_students, seed=seed).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    write_synthetic_data("data/synthetic/student_success.csv")
