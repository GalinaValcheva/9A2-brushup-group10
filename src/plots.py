from pathlib import Path

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay



OUTPUT_DIR = Path("outputs")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_training_curve(history_path, output_prefix, title_prefix):
    if not history_path.exists():
        print(f"Skipping {history_path}; file does not exist.")
        return

    history_df = pd.read_csv(history_path)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} training and validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_prefix}_loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_auc"], label="train AUC")
    plt.plot(history_df["epoch"], history_df["val_auc"], label="validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"{title_prefix} training and validation AUC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_prefix}_auc_curve.png", dpi=150)
    plt.close()


def plot_confusion_matrices_from_metrics():
    result_files = [
        OUTPUT_DIR / "baseline_metrics.json",
        OUTPUT_DIR / "cnn_test_metrics.json",
    ]

    weighted_path = OUTPUT_DIR / "cnn_weighted_test_metrics.json"
    if weighted_path.exists():
        result_files.append(weighted_path)

    fig, axes = plt.subplots(1, len(result_files), figsize=(5 * len(result_files), 4))

    if len(result_files) == 1:
        axes = [axes]

    for ax, result_file in zip(axes, result_files):
        result = load_json(result_file)
        cm = np.array(result["test_metrics"]["confusion_matrix"])

        ConfusionMatrixDisplay(confusion_matrix=cm).plot(
            ax=ax,
            colorbar=False,
        )
        ax.set_title(result["model"])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    plot_training_curve(
        OUTPUT_DIR / "cnn_training_log.csv",
        "cnn",
        "CNN"
    )

    plot_training_curve(
        OUTPUT_DIR / "cnn_weighted_training_log.csv",
        "cnn_weighted",
        "Class-weighted CNN"
    )

    plot_confusion_matrices_from_metrics()

    print("Saved plots to outputs/")


if __name__ == "__main__":
    main()