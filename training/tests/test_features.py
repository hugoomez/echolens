import numpy as np

from echolens.features.melspec import CLIP_SAMPLES, N_MELS, SR, wav_to_logmel


def test_logmel_output_shape() -> None:
    """A silent clip should still produce a log-mel array with N_MELS rows."""
    y = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    logmel = wav_to_logmel(y)
    assert logmel.shape[0] == N_MELS
    assert logmel.dtype == np.float32


def test_pure_tone_energy_lands_in_expected_mel_band() -> None:
    """A pure sine tone should produce its strongest energy in a plausible mel
    band, not at the very top or bottom of the mel range."""
    freq = 1000.0  # Hz
    t = np.arange(CLIP_SAMPLES) / SR
    y = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    logmel = wav_to_logmel(y)
    band_energy = logmel.mean(axis=1)
    loudest_band = int(np.argmax(band_energy))

    # 1000 Hz should land in the lower-middle part of the mel scale
    assert 5 <= loudest_band <= 35
