"""Evaluate the saved model and write the final overfitting analysis."""
from pathlib import Path

import matplotlib
# Save images directly instead of opening GUI windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from train_model import StudentPerformanceNet, metrics


def conclusion(history, train_acc, test_acc, train_loss, test_loss):
    """Use accuracy gap plus curve behavior, not hard-coded result values."""
    gap = train_acc - test_acc
    min_test_loss = min(history["test_loss"])
    min_epoch = history["test_loss"].index(min_test_loss) + 1
    rising_test_loss = test_loss > min_test_loss + 0.05
    training_kept_improving = train_loss < history["train_loss"][min_epoch - 1]
    if gap >= 10 and rising_test_loss and training_kept_improving:
        return ("Yes. The behavior is consistent with overfitting: the model fits the "
                "training students much better than unseen students, while test loss rose "
                "after its best value as training loss continued to decrease."), min_test_loss, min_epoch
    return ("No clear overfitting was detected by the selected criteria. The training and "
            "unseen-test results did not show a sufficiently large gap together with worsening test loss."), min_test_loss, min_epoch


def main():
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs"
    checkpoint = torch.load(outputs / "student_model.pt", map_location="cpu", weights_only=False)
    data = pd.read_csv(root / "data" / "students.csv")
    features = checkpoint["features"]
    train_idx, test_idx = checkpoint["train_indices"], checkpoint["test_indices"]
    mean, scale = checkpoint["scaler_mean"], checkpoint["scaler_scale"]
    x_all = data[features].to_numpy()
    x_train = (x_all[train_idx] - mean) / scale
    x_test = (x_all[test_idx] - mean) / scale
    y_all = data["performance"].to_numpy()
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = StudentPerformanceNet(len(features))
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc = metrics(model, x_train_t, y_train_t, criterion)
    test_loss, test_acc = metrics(model, x_test_t, y_test_t, criterion)
    history = checkpoint["history"]
    final_conclusion, best_test_loss, best_epoch = conclusion(
        history, train_acc, test_acc, train_loss, test_loss
    )

    model.eval()
    with torch.no_grad():
        predictions = model(x_test_t).argmax(dim=1).numpy()
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["At Risk", "Good Performance"],
                yticklabels=["At Risk", "Good Performance"])
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Confusion Matrix: Unseen Test Students")
    plt.tight_layout()
    plt.savefig(outputs / "confusion_matrix.png", dpi=150)
    plt.close()

    gap = train_acc - test_acc
    text = f"""OVERFITTING ANALYSIS

Training Accuracy: {train_acc:.2f}%
Test Accuracy: {test_acc:.2f}%

Training Loss: {train_loss:.4f}
Test Loss: {test_loss:.4f}

Generalization Gap: {gap:.2f} percentage points

Best Test Loss: {best_test_loss:.4f} (Epoch {best_epoch})

Conclusion:
{final_conclusion}
"""
    (outputs / "results.txt").write_text(text)
    print(text)
    print("Confusion matrix saved to outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
