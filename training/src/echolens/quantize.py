import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


class MelCalibrationReader(CalibrationDataReader):
    """Feeds representative log-mel spectrograms to the quantizer so it
    can estimate realistic value ranges (calibration), instead of guessing."""

    def __init__(self, logmels: list[np.ndarray], input_name: str = "logmel"):
        self.data = iter([{input_name: m[None, None].astype(np.float32)} for m in logmels])

    def get_next(self):
        return next(self.data, None)


def quantize(in_path: str, out_path: str, calib_logmels: list[np.ndarray]) -> None:
    """Static INT8 quantization with QDQ format and per-channel weights."""
    quantize_static(
        in_path,
        out_path,
        calibration_data_reader=MelCalibrationReader(calib_logmels),
        quant_format=QuantFormat.QDQ,  # recommended for CNNs
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
