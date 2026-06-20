import librosa
import numpy as np

# --- Frozen feature contract ---
# These constants define exactly how the log-mel spectrogram is computed.
# They must match between training (Python) and inference (JavaScript).
# Changing any of these requires re-training and re-exporting everything.
SR = 22050  # sample rate (Hz)
N_FFT = 1024  # FFT window size (samples)
HOP = 512  # hop length between windows (samples), 50% overlap with N_FFT
N_MELS = 64  # number of mel frequency bands
FMIN, FMAX = 0, SR // 2  # frequency range covered by the mel filterbank
CLIP_SAMPLES = SR * 4  # all clips are normalized to exactly 4 seconds

# Mel filterbank matrix (Slaney normalization).
# Computed once at import time. This exact matrix is exported to JSON
# so the browser reuses it instead of recomputing its own (and risking
# small numerical mismatches).
MEL_FB = librosa.filters.mel(
    sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, htk=False, norm="slaney"
)  # shape: (N_MELS, N_FFT//2 + 1)


def export_filterbank(path: str) -> None:
    """Dump the mel filterbank matrix and its parameters to JSON,
    so the browser-side feature extractor can load and reuse it directly."""
    import json

    with open(path, "w") as f:
        json.dump(
            {"sr": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "filterbank": MEL_FB.tolist()},
            f,
        )


def wav_to_logmel(y: np.ndarray) -> np.ndarray:
    """Convert a mono float32 waveform into a log-mel spectrogram, shape [N_MELS, T]."""
    # Pad with silence (or truncate) so every clip has the same fixed length
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    else:
        y = y[:CLIP_SAMPLES]

    # Short-Time Fourier Transform: decompose the signal into frequencies over time
    stft = librosa.stft(
        y, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window="hann", center=True
    )

    # Power spectrum: magnitude squared, discard phase information
    power = np.abs(stft) ** 2.0

    # Project onto the mel scale via matrix multiplication with the precomputed filterbank
    mel = MEL_FB @ power  # (N_MELS, T)

    # Log compression (mimics human perception of loudness); +1e-6 avoids log(0)
    return np.log(mel + 1e-6).astype(np.float32)
