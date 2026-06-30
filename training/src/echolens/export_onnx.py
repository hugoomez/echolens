import torch

from echolens.models.cnn import AudioCNN


def export_fp32(ckpt_path: str, out_path: str, n_mels: int = 64, t: int = 173) -> None:
    """Export a trained AudioCNN checkpoint to ONNX (FP32).

    t is just a placeholder time-frame size used to trace the graph;
    the actual frame axis is declared dynamic, so the real model will
    accept any number of frames at inference time.
    """
    model = AudioCNN(n_classes=10)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 1, n_mels, t)  # (batch, channels, mel_bands, frames)
    torch.onnx.export(
        model,
        (dummy,),
        out_path,
        input_names=["logmel"],
        output_names=["logits"],
        opset_version=18,
        dynamic_axes={
            "logmel": {0: "batch", 3: "frames"},  # batch and frame count are variable
            "logits": {0: "batch"},
        },
    )
