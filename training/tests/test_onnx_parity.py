import os

import numpy as np
import onnxruntime as ort
import pytest
import torch

from echolens.models.cnn import AudioCNN

FIXTURE_PATH = "../web/tests/fixtures/parity_reference.npz"
CHECKPOINT_PATH = "checkpoint_best.pt"
ONNX_PATH = "../model_fp32.onnx"

requires_artifacts = pytest.mark.skipif(
    not (
        os.path.exists(FIXTURE_PATH)
        and os.path.exists(CHECKPOINT_PATH)
        and os.path.exists(ONNX_PATH)
    ),
    reason="Trained checkpoint, ONNX export or parity fixture not available in this environment.",
)


@requires_artifacts
def test_pytorch_vs_onnx() -> None:
    """The exported ONNX model must produce logits numerically equivalent
    to the original PyTorch model, on the same reference input."""
    ref = np.load(FIXTURE_PATH)
    x = ref["logmel"][None, None].astype(np.float32)  # (1, 1, 64, T)

    # PyTorch
    model = AudioCNN(n_classes=10)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        y_torch = model(torch.from_numpy(x)).numpy()

    # ONNX
    sess = ort.InferenceSession(ONNX_PATH)
    y_onnx = sess.run(["logits"], {"logmel": x})[0]

    np.testing.assert_allclose(y_torch, y_onnx, rtol=1e-3, atol=1e-4)
