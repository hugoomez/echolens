import numpy as np


def spec_augment(
    logmel: np.ndarray,
    freq_mask_param: int = 8,
    time_mask_param: int = 16,
    n_freq_masks: int = 2,
    n_time_masks: int = 2,
) -> np.ndarray:
    """Apply SpecAugment: random frequency and time masking on a log-mel spectrogram.

    logmel: array of shape (n_mels, T)
    Returns a copy with masked regions filled with the spectrogram's mean value.
    """
    augmented = logmel.copy()
    n_mels, n_frames = augmented.shape
    mask_value = augmented.mean()

    # Frequency masking: blank out a few contiguous mel bands
    for _ in range(n_freq_masks):
        f_width = np.random.randint(0, freq_mask_param + 1)
        if f_width == 0 or f_width >= n_mels:
            continue
        f_start = np.random.randint(0, n_mels - f_width)
        augmented[f_start : f_start + f_width, :] = mask_value

    # Time masking: blank out a few contiguous time frames
    for _ in range(n_time_masks):
        t_width = np.random.randint(0, time_mask_param + 1)
        if t_width == 0 or t_width >= n_frames:
            continue
        t_start = np.random.randint(0, n_frames - t_width)
        augmented[:, t_start : t_start + t_width] = mask_value

    return augmented


def mixup(x, y, alpha: float = 0.2):
    """Batch-level mixup: blends pairs of samples and their labels.

    Returns the mixed inputs, the two sets of original labels, and the
    mixing coefficient lambda, so the loss can be computed as a weighted
    combination of both labels. Used in the training loop (step 1.8).
    """
    import torch

    batch_size = x.size(0)
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[perm]
    return mixed_x, y, y[perm], lam
