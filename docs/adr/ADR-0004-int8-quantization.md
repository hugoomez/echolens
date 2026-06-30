# ADR-0004: ONNX export opset and quantization decision

## Status
Accepted

## Context
The Phase 2 plan specifies ONNX opset 17 for export, and static INT8
quantization (QDQ format, per-channel) to reduce model size and
potentially speed up inference, expecting roughly a 4x size reduction
with <2 accuracy points lost.

## Decision 1 — Export opset
Use **opset 18** instead of 17. PyTorch's exporter (dynamo-based, the
current default) targets opset 18 internally; downconverting the
exported graph to opset 17 fails with an internal ONNX version-converter
error (`axes_input_to_attribute` assertion). Opset 18 is fully supported
by `onnxruntime-web`, so there is no practical downside.

## Decision 2 — Do NOT ship the INT8 quantized model
After implementing static INT8 quantization (QDQ format, per-channel,
200 calibration samples from the training folds) and measuring the
actual results, we decided to **keep only the FP32 model in production**
and drop INT8.

### Measured results

| Model | Size | Accuracy (fold 10, ONNX Runtime) |
|-------|------|-----------------------------------|
| FP32  | 36 KB  | 0.8614 |
| INT8  | 330 KB | 0.8519 |

INT8 quantization made the model **~9x larger**, not smaller, and lost
0.96 accuracy points — the opposite of the expected outcome.

### Why this happened
`AudioCNN` (Phase 1) is intentionally tiny: few channels per block
(32/64/128), only 3 convolutional blocks, and `AdaptiveAvgPool2d(1)`
before the final linear layer, which keeps that layer to ~1280
parameters instead of the tens of thousands a flattened version would
need. The resulting FP32 model is only 36 KB — smaller than most web
page icons.

QDQ-format quantization inserts Quantize/DeQuantize nodes and
per-channel scale/zero-point metadata throughout the graph. For a large
model, this overhead is negligible next to the savings from storing
millions of weights in 8 bits instead of 32. For a model this small, the
overhead outweighs the savings entirely — there are too few weights for
INT8 storage to pay off, and the extra graph nodes dominate the file
size instead.

Additionally, the original justification for quantizing — reducing
download size and/or speeding up browser inference — does not apply
here: a 36 KB download is already negligible, and INT8 speedups are not
guaranteed on hardware lacking VNNI instructions (see the project's risk
register, R8), so there was no real bottleneck to solve in the first
place.

## Alternatives considered
- `QuantFormat.QOperator` instead of QDQ: would likely reduce the
  overhead somewhat, but was not pursued once the accuracy-vs-size
  tradeoff was already shown to be unfavorable for this model size.
- Dynamic quantization: skipped per the plan's own guidance that static
  calibration-based quantization gives better accuracy for CNNs; this
  would not have addressed the size regression at this model size.

## Consequences
- Only `web/public/models/model.onnx` (FP32) ships to the browser.
  No INT8 variant is produced or maintained.
- Phase 5 (benchmarks) does not need to compare FP32 vs INT8 inference
  speed, since INT8 was not adopted.
- If the model architecture grows significantly in the future (more
  channels, more classes), INT8 quantization should be re-evaluated, as
  the tradeoff that ruled it out here is a direct consequence of how
  small this specific model is.