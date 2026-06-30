import os

import pytest

from echolens.evaluate_onnx import evaluate_onnx
from pathlib import Path

FP32_PATH = str(Path(__file__).resolve().parents[2] / "model_fp32.onnx")

requires_artifacts = pytest.mark.skipif(
    not os.path.exists(FP32_PATH),
    reason="FP32 ONNX export not available in this environment.",
)


@requires_artifacts
def test_fp32_accuracy_matches_expected_baseline() -> None:
    """The shipped FP32 ONNX model must reproduce the accuracy measured
    during Phase 2 (within a small tolerance), on the held-out fold 10.

    Note: INT8 quantization was evaluated and rejected (see ADR-0004) —
    it made this model ~9x larger with a measurable accuracy drop,
    since the model is already extremely small. Only FP32 ships.
    """
    acc = evaluate_onnx(FP32_PATH, fold=10)
    assert acc >= 0.80  # consistent with the Phase 1 acceptance criterion
