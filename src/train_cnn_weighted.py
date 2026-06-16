from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from data import get_torch_loaders
from metrics import compute_binary_metrics
from models import SmallCNN
from utils import save_json, set_seed


SEED = 123
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_one_epoch(model, loader, criterion, device, optimizer=None, training=False):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = torch.argmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)

        all_labels.extend(labels.detach().cpu().numpy())
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_probabilities.extend(probabilities.detach().cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    metrics = compute_binary_metrics(all_labels, all_predictions, all_probabilities)
    metrics["loss"] = float(average_loss)

    return metrics


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_torch_loaders(
        batch_size=BATCH_SIZE
    )

    model = SmallCNN(num_classes=2).to(device)

    dummy_input = torch.randn(4, 1, 28, 28).to(device)
    dummy_output = model(dummy_input)
    print("Dummy input shape:", dummy_input.shape)
    print("Dummy output shape:", dummy_output.shape)

    labels = train_dataset.labels.squeeze()
    class_counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
    class_weights = len(labels) / (2.0 * class_counts.float())
    class_weights = class_weights.to(device)

    print("Class counts:", class_counts)
    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_auc = -1
    history = []
    checkpoint_path = OUTPUT_DIR / "best_cnn_weighted_model.pt"

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")

        train_metrics = run_one_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            training=True
        )

        val_metrics = run_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            training=False
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_auc": train_metrics["auc"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_f1": val_metrics["f1"],
        }

        history.append(row)
        print(row)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_auc": best_val_auc,
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "model": "SmallCNNWeightedLoss",
            }, checkpoint_path)

            print("Saved new best checkpoint")

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "cnn_weighted_training_log.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    best_model = SmallCNN(num_classes=2).to(device)
    best_model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_one_epoch(
        best_model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        training=False
    )

    save_json({
        "model": "small_cnn_class_weighted",
        "checkpoint": str(checkpoint_path),
        "best_validation_auc": checkpoint["best_val_auc"],
        "test_metrics": test_metrics,
    }, OUTPUT_DIR / "cnn_weighted_test_metrics.json")

    print("Best validation AUC:", checkpoint["best_val_auc"])
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()