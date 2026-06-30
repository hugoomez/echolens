import numpy as np
import onnxruntime as ort

ref = dict(np.load("web/tests/fixtures/parity_reference.npz"))

sess = ort.InferenceSession("web/public/models/model.onnx")
x = ref["logmel"][None, None].astype(np.float32)
ref["expected_logits"] = sess.run(["logits"], {"logmel": x})[0]

np.savez("web/tests/fixtures/parity_reference.npz", **ref)
print("parity_reference.npz now contains:", list(ref.keys()))
