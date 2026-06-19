import csv
import json
import os
import random

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from echolens.data.dataset import UrbanSound8K
from echolens.models.cnn import AudioCNN

ALL_FOLDS = list(range(1, 11))


def set_seed(seed: int) -> None:
    """Fix all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_norm_stats(folds: list[int], cache_dir: str) -> tuple[float, float]:
    """Global mean/std of log-mel values over the given folds, without augmentation."""
    ds = UrbanSound8K(folds=folds, cache_dir=cache_dir, augment=False)
    total_sum, total_sq_sum, total_count = 0.0, 0.0, 0
    for i in range(len(ds)):
        x, _ = ds[i]
        x_np = x.numpy()
        total_sum += x_np.sum()
        total_sq_sum += (x_np**2).sum()
        total_count += x_np.size
    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean**2
    std = float(np.sqrt(max(var, 1e-12)))
    return float(mean), std


def normalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x - mean) / (std + 1e-8)


def run_epoch(model, loader, device, mean, std, optimizer=None, mixup_alpha: float = 0.0):
    """Run one epoch. Trains if optimizer is given, otherwise evaluates."""
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = normalize(x, mean, std)

        if is_train and mixup_alpha > 0:
            from echolens.features.augment import mixup

            mixed_x, y_a, y_b, lam = mixup(x, y, alpha=mixup_alpha)
            logits = model(mixed_x)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        else:
            logits = model(x)
            loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += x.size(0)

    return total_loss / total_count, total_correct / total_count


def train_one_split(train_folds, val_folds, cfg: DictConfig, log_path: str | None = None):
    """Train on train_folds, evaluate on val_folds. Returns (best_val_acc, model, mean, std)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = compute_norm_stats(train_folds, cfg.data.cache_dir)

    train_ds = UrbanSound8K(train_folds, cfg.data.cache_dir, augment=cfg.train.augment)
    val_ds = UrbanSound8K(val_folds, cfg.data.cache_dir, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    model = AudioCNN(n_classes=cfg.model.n_classes).to(device)
    print(f"[device check] training device = {device}")
    print(f"[device check] model parameters on = {next(model.parameters()).device}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    best_val_acc = 0.0
    best_state = None
    log_file = open(log_path, "w", newline="") if log_path else None
    writer = csv.writer(log_file) if log_file else None
    if writer:
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(cfg.train.epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, device, mean, std, optimizer, mixup_alpha=0.2
        )
        val_loss, val_acc = run_epoch(model, val_loader, device, mean, std)
        scheduler.step()
        print(
            f"  epoch {epoch + 1}/{cfg.train.epochs} — train_loss={train_loss:.3f} train_acc={train_acc:.3f} val_loss={val_loss:.3f} val_acc={val_acc:.3f}"
        )

        if writer:
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if log_file:
        log_file.close()
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc, model, mean, std


@hydra.main(config_path="../../configs", config_name="baseline", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs("training/logs", exist_ok=True)

    # --- Honest 10-fold cross-validation metric ---
    fold_accuracies = []
    for k in ALL_FOLDS:
        train_folds = [f for f in ALL_FOLDS if f != k]
        acc, _, _, _ = train_one_split(
            train_folds, [k], cfg, log_path=f"training/logs/fold_{k}.csv"
        )
        print(f"Fold {k}: val accuracy = {acc:.4f}")
        fold_accuracies.append(acc)

    mean_acc = float(np.mean(fold_accuracies))
    print(f"10-fold CV mean accuracy: {mean_acc:.4f}")
    with open("training/logs/cv_results.json", "w") as f:
        json.dump({"fold_accuracies": fold_accuracies, "mean_accuracy": mean_acc}, f, indent=2)

    # --- Final deployable model: train on folds 1-9, validate on fold 10 ---
    final_acc, final_model, mean, std = train_one_split(
        list(range(1, 10)), [10], cfg, log_path="training/logs/final_model.csv"
    )
    print(f"Final model accuracy (fold 10): {final_acc:.4f}")

    torch.save(final_model.state_dict(), "training/checkpoint_best.pt")
    with open("web/public/models/norm_stats.json", "w") as f:
        json.dump({"mean": mean, "std": std}, f)


if __name__ == "__main__":
    main()
