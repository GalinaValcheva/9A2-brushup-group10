import json
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    baseline = load_json(OUTPUT_DIR / "baseline_metrics.json")
    cnn = load_json(OUTPUT_DIR / "cnn_test_metrics.json")
    cnn_weighted = load_json(OUTPUT_DIR / "ccn_weighted_test_metrics.json")

    rows = []

    for result in [baseline, cnn, cnn_weighted]:
        metrics = result["test_metrics"]

        rows.append({
            "model": result["model"],
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_DIR / "final_model_comparison.csv", index=False)

    print(comparison)


if __name__ == "__main__":
    main()