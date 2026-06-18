import json
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def add_result_row(rows, result):
    metrics = result["test_metrics"]

    rows.append({
        "model": result["model"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
    })


def main():
    result_files = [
        OUTPUT_DIR / "baseline_metrics.json",
        OUTPUT_DIR / "cnn_test_metrics.json",
    ]

    weighted_path = OUTPUT_DIR / "cnn_weighted_test_metrics.json"
    if weighted_path.exists():
        result_files.append(weighted_path)

    rows = []

    for result_file in result_files:
        result = load_json(result_file)
        add_result_row(rows, result)

    comparison = pd.DataFrame(rows)
    Path("experiments").mkdir(exist_ok=True)
    comparison.to_csv("experiments/final_model_comparison.csv", index=False)
    #comparison.to_csv(OUTPUT_DIR / "final_model_comparison.csv", index=False)

    print(comparison)


if __name__ == "__main__":
    main()
