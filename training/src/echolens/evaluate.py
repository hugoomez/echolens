import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from echolens.data.dataset import UrbanSound8K
from echolens.models.cnn import AudioCNN
from echolens.train import normalize

CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("web/public/models/norm_stats.json") as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    model = AudioCNN(n_classes=10).to(device)
    model.load_state_dict(torch.load("training/checkpoint_best.pt", map_location=device))
    model.eval()

    # Fold 10 was never seen during training of the final model — honest held-out evaluation
    val_ds = UrbanSound8K(folds=[10], cache_dir="training/data/cache", augment=False)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = normalize(x.to(device), mean, std)
            preds = model(x).argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    with open("training/logs/cv_results.json") as f:
        cv = json.load(f)

    # --- Confusion matrix plot ---
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — final model (fold 10 held-out)")
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("docs/confusion_matrix.png", dpi=150)

    # --- metrics.md ---
    lines = [
        "## Baseline model (AudioCNN) — UrbanSound8K, 10-fold CV\n",
        "| Metric | FP32 |",
        "|--------|------|",
        f"| Mean accuracy (10-fold CV) | {cv['mean_accuracy']:.4f} |",
        f"| Macro-F1 (fold 10, final model) | {macro_f1:.4f} |\n",
        "### Per-class accuracy (fold 10, final model)",
        "| Class | Acc |",
        "|-------|-----|",
    ]
    for name, acc in zip(CLASS_NAMES, per_class_acc):
        lines.append(f"| {name} | {acc:.4f} |")
    lines += [
        "\n![Confusion matrix](confusion_matrix.png)\n",
        "### Accuracy per fold (10-fold CV)",
        "| Fold | Acc |",
        "|------|-----|",
    ]
    for i, acc in enumerate(cv["fold_accuracies"], start=1):
        lines.append(f"| {i} | {acc:.4f} |")

    with open("docs/metrics.md", "w") as f:
        f.write("\n".join(lines))

    print(f"Macro-F1 (fold 10): {macro_f1:.4f}")
    print("Wrote docs/metrics.md and docs/confusion_matrix.png")


if __name__ == "__main__":
    main()
