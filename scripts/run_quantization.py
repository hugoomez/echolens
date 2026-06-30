import random

from echolens.data.dataset import UrbanSound8K
from echolens.quantize import quantize

# Calibration data must come from the training set, never from validation/test
ds = UrbanSound8K(
    folds=list(range(1, 10)), cache_dir="training/data/cache", augment=False
)

random.seed(42)
indices = random.sample(range(len(ds)), k=200)
calib_logmels = [ds[i][0].squeeze(0).numpy() for i in indices]

quantize("model_prep.onnx", "web/public/models/model.onnx", calib_logmels)
print(f"Quantized model written using {len(calib_logmels)} calibration samples.")
