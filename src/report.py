"""Generate a deterministic, reviewable evaluation report."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .baseline import SENSITIVE_COLUMNS, TARGET, train_baseline
from .data_quality import assert_valid_student_data
from .evaluation import fairness_summary, subgroup_metrics


def generate_report(data_path: str | Path, output_dir: str | Path, seed: int = 42) -> Path:
    data = pd.read_csv(data_path)
    quality = assert_valid_student_data(data)
    _, results, metrics = train_baseline(data, seed=seed)
    fairness = {}
    for column in SENSITIVE_COLUMNS:
        table = subgroup_metrics(results, column, TARGET, "prediction")
        fairness[column] = fairness_summary(table)
    payload = {
        "data": str(data_path),
        "seed": seed,
        "quality": quality.as_dict(),
        "metrics": metrics,
        "fairness": fairness,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "evaluation_report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, help="JSON experiment config")
    args = parser.parse_args()
    seed = args.seed
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        seed = int(config.get("seed", seed))
    print(generate_report(args.data, args.output_dir, seed))


if __name__ == "__main__":
    main()
