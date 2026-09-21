# Student Success Data Analytics

This repository reviews the main models used to analyze and improve student
success. It brings together model definitions, assumptions, evaluation
approaches, and practical guidance for understanding which factors are
associated with student outcomes.

## Focus areas

- Student retention and persistence
- Graduation and completion likelihood
- Academic performance and early-alert models
- Student engagement and support needs
- Equity, fairness, interpretability, and responsible use

The goal is to make student success analytics easier to understand, compare,
validate, and apply responsibly in educational settings.

## Quick start

The repository uses synthetic data only. No real student records should be
committed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m src.baseline --data data/synthetic/student_success.csv
pytest
```

The reproducible baseline notebook is
[`notebooks/01_baseline_student_success.ipynb`](notebooks/01_baseline_student_success.ipynb).
It predicts whether a student will persist to the next academic term and
reports performance, calibration, and subgroup fairness metrics.

## Repository structure

| Path | Purpose |
| --- | --- |
| `data/synthetic/` | Reproducible, non-sensitive demonstration data |
| `src/` | Data generation, modeling, evaluation, and fairness utilities |
| `notebooks/` | Reproducible analyses |
| `docs/` | Data, evaluation, and responsible-use documentation |
| `tests/` | Automated unit tests |
| `.github/workflows/` | Continuous integration |

## Responsible use

This project is educational and exploratory. Model outputs must not be used
as automated decisions about students. Read
[`docs/responsible_use.md`](docs/responsible_use.md) before adapting the
workflow to any real setting.
