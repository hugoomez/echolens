import json

import numpy as np
import onnxruntime as ort

from echolens.data.dataset import UrbanSound8K
from echolens.train import normalize
from pathlib import Path


def evaluate_onnx(onnx_path: str, fold: int = 10) -> float:
    repo_root = Path(__file__).resolve().parents[3]
    norm_stats_path = repo_root / "web" / "public" / "models" / "norm_stats.json"
    with open(norm_stats_path) as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    ds = UrbanSound8K(
        folds=[fold], cache_dir=str(repo_root / "training" / "data" / "cache"), augment=False
    )
    sess = ort.InferenceSession(onnx_path)

    correct, total = 0, 0
    for i in range(len(ds)):
        x, y = ds[i]
        x = normalize(x.unsqueeze(0), mean, std).numpy().astype(np.float32)
        logits = sess.run(["logits"], {"logmel": x})[0]
        pred = int(np.argmax(logits))
        correct += int(pred == y)
        total += 1

    return correct / total


if __name__ == "__main__":
    fp32_acc = evaluate_onnx("model_fp32.onnx")
    int8_acc = evaluate_onnx("web/public/models/model.onnx")
    print(f"FP32 (ONNX) accuracy on fold 10: {fp32_acc:.4f}")
    print(f"INT8 (ONNX) accuracy on fold 10: {int8_acc:.4f}")
    print(f"Accuracy drop: {(fp32_acc - int8_acc) * 100:.2f} points")
