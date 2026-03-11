import argparse
import json
import os
import random
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Allow running both `python src/train.py` and `python -m src.train`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ResNet1D_12Lead, count_parameters


CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NpyEcgDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 3 or X.shape[1:] != (12, 1000):
            raise ValueError(f"Expected X shape (N, 12, 1000), got {X.shape}")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"Expected y shape (N,), got {y.shape} vs X {X.shape}")
        # Convert once to torch tensors for faster CPU training.
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Some classes have zero samples in training set: {counts}")
    inv = 1.0 / counts
    w = inv / inv.sum() * num_classes  # normalized to sum to num_classes
    return torch.tensor(w, dtype=torch.float32)


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    val_acc: float
    val_macro_f1: float


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    losses = []
    all_y = []
    all_logits = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        all_logits.append(logits.detach().cpu())
        all_y.append(yb.detach().cpu())
    y_true = torch.cat(all_y).numpy()
    logits = torch.cat(all_logits).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_pred = probs.argmax(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))

    # AUC per class (OvR)
    auc_per_class = roc_auc_score(y_true, probs, multi_class="ovr", average=None)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "auc_per_class": auc_per_class,
    }


def plot_training_curves(history: list[EpochMetrics], out_path: str) -> None:
    epochs = np.arange(1, len(history) + 1)
    train_loss = [h.train_loss for h in history]
    val_loss = [h.val_loss for h in history]
    val_f1 = [h.val_macro_f1 for h in history]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    ax[0].plot(epochs, train_loss, label="train")
    ax[0].plot(epochs, val_loss, label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Cross-entropy")
    ax[0].legend(frameon=False)

    ax[1].plot(epochs, val_f1, label="val macro F1")
    ax[1].set_title("Validation macro F1")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("F1")
    ax[1].set_ylim(0, 1)
    ax[1].legend(frameon=False)

    fig.suptitle("Training Curves", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="PHASE 2: Train ResNet1D_12Lead on PTB-XL superclasses.")
    parser.add_argument("--results-dir", type=str, default="outputs/results", help="Directory containing X_*.npy/y_*.npy")
    parser.add_argument("--models-dir", type=str, default="outputs/models", help="Where to save checkpoints")
    parser.add_argument("--figures-dir", type=str, default="outputs/figures", help="Where to save figures")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    if os.cpu_count() is not None:
        torch.set_num_threads(min(8, os.cpu_count()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | torch num_threads={torch.get_num_threads()}", flush=True)

    # Load into memory for faster CPU training (avoids per-sample copy overhead).
    X_train = np.load(os.path.join(args.results_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.results_dir, "y_train.npy"))
    X_val = np.load(os.path.join(args.results_dir, "X_val.npy"))
    y_val = np.load(os.path.join(args.results_dir, "y_val.npy"))

    train_ds = NpyEcgDataset(X_train, y_train)
    val_ds = NpyEcgDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = 5
    class_weights = compute_class_weights(y_train, num_classes=num_classes).to(device)
    print("Class weights (sum=5):", class_weights.detach().cpu().numpy(), flush=True)

    model = ResNet1D_12Lead(num_classes=num_classes, kernel_size=7, dropout=0.3).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    best_f1 = -1.0
    best_epoch = -1
    patience = 10
    epochs_no_improve = 0
    history: list[EpochMetrics] = []

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    ckpt_path = os.path.join(args.models_dir, "ResNet1D_12Lead_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        n_batches = len(train_loader)
        for b_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)
            n_seen += int(xb.size(0))
            if b_idx % 50 == 0 or b_idx == n_batches:
                print(f"  epoch {epoch:02d} train batch {b_idx:03d}/{n_batches}", flush=True)
        train_loss = running_loss / max(n_seen, 1)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_running_loss += float(loss.item()) * xb.size(0)
                val_seen += int(xb.size(0))
        val_loss = val_running_loss / max(val_seen, 1)

        eval_out = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_acc = eval_out["accuracy"]
        val_macro_f1 = eval_out["macro_f1"]

        history.append(EpochMetrics(train_loss=train_loss, val_loss=val_loss, val_acc=val_acc, val_macro_f1=val_macro_f1))

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | lr={lr_now:.2e} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_macro_f1={val_macro_f1:.4f}"
        , flush=True)

        scheduler.step(val_macro_f1)

        improved = val_macro_f1 > best_f1 + 1e-6
        if improved:
            best_f1 = val_macro_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_name": "ResNet1D_12Lead",
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "classes": CLASSES,
                    "seed": args.seed,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping: no val macro F1 improvement for {patience} epochs.", flush=True)
            break

    plot_training_curves(history, os.path.join(args.figures_dir, "training_curves.png"))

    # Load best checkpoint and report detailed validation metrics
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["state_dict"])
    eval_out = evaluate(model, val_loader, device=device, num_classes=num_classes)

    print()
    print(f"Best checkpoint: epoch={best['epoch']} val_macro_f1={best['val_macro_f1']:.4f}", flush=True)
    print(f"VAL accuracy:  {eval_out['accuracy']:.4f}", flush=True)
    print(f"VAL macro F1:  {eval_out['macro_f1']:.4f}", flush=True)
    print("VAL per-class F1:", flush=True)
    for c, f1 in zip(CLASSES, eval_out["per_class_f1"]):
        print(f"  {c:4s}: {float(f1):.4f}", flush=True)
    print("VAL AUC per class:", flush=True)
    for c, auc in zip(CLASSES, eval_out["auc_per_class"]):
        print(f"  {c:4s}: {float(auc):.4f}", flush=True)

    # Save a tiny json for quick inspection (kept in results/ as it's text)
    train_summary_path = os.path.join(args.results_dir, "training_summary.json")
    with open(train_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "ResNet1D_12Lead",
                "device": str(device),
                "best_epoch": int(best_epoch),
                "best_val_macro_f1": float(best_f1),
                "val_accuracy": float(eval_out["accuracy"]),
                "val_macro_f1": float(eval_out["macro_f1"]),
                "val_auc_per_class": {c: float(a) for c, a in zip(CLASSES, eval_out["auc_per_class"])},
            },
            f,
            indent=2,
        )
    print(f"Saved: {ckpt_path}", flush=True)
    print(f"Saved: {os.path.join(args.figures_dir, 'training_curves.png')}", flush=True)
    print(f"Saved: {train_summary_path}", flush=True)


if __name__ == "__main__":
    main()

