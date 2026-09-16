"""Train a small PyTorch network and save metrics/learning curves."""
from pathlib import Path
import random

import matplotlib
# A non-interactive backend lets the scripts save figures in VS Code/macOS
# terminals without requiring a GUI window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

SEED = 42
FEATURES = [
    "study_hours", "attendance", "previous_score", "assignment_score",
    "online_learning_hours", "quiz_score", "sleep_hours",
]
EPOCHS = 500
LEARNING_RATE = 0.01


def set_seed(seed=SEED):
    """Make splitting, initialization, and training repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class StudentPerformanceNet(nn.Module):
    """A simple feed-forward classifier: 7 inputs -> 64 -> 32 -> 2 classes."""
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.layers(x)


def metrics(model, x, y, criterion):
    """Evaluate without changing model weights."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y).item()
        accuracy = (logits.argmax(dim=1) == y).float().mean().item() * 100
    return loss, accuracy


def save_curves(history, outputs_dir):
    epochs = range(1, EPOCHS + 1)
    plt.style.use("seaborn-v0_8-whitegrid")
    for metric, ylabel, filename in [
        ("loss", "Cross-Entropy Loss", "loss_curve.png"),
        ("accuracy", "Accuracy (%)", "accuracy_curve.png"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history[f"train_{metric}"], label="Training", linewidth=2)
        plt.plot(epochs, history[f"test_{metric}"], label="Unseen test", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"Training vs Test {ylabel}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outputs_dir / filename, dpi=150)
        plt.close()


def main():
    set_seed()
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs"
    outputs.mkdir(exist_ok=True)
    data = pd.read_csv(root / "data" / "students.csv")
    x = data[FEATURES]
    y = data["performance"]

    # The scaler learns only from training students; test students remain unseen.
    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.20, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x.iloc[train_idx])
    x_test = scaler.transform(x.iloc[test_idx])
    y_train = y.iloc[train_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = StudentPerformanceNet(len(FEATURES))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {key: [] for key in ("train_loss", "test_loss", "train_accuracy", "test_accuracy")}

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        train_loss, train_acc = metrics(model, x_train_t, y_train_t, criterion)
        test_loss, test_acc = metrics(model, x_test_t, y_test_t, criterion)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_acc)
        history["test_accuracy"].append(test_acc)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{EPOCHS} | Train loss: {train_loss:.4f} | "
                  f"Test loss: {test_loss:.4f} | Train acc: {train_acc:.2f}% | Test acc: {test_acc:.2f}%")

    save_curves(history, outputs)
    checkpoint = {
        "model_state": model.state_dict(), "features": FEATURES,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "train_indices": train_idx, "test_indices": test_idx, "history": history,
    }
    torch.save(checkpoint, outputs / "student_model.pt")
    print("\nSaved trained model and learning curves in outputs/.")


if __name__ == "__main__":
    main()
