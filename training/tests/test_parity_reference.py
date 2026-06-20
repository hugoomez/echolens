import os

import numpy as np
import pytest

from echolens.data.dataset import UrbanSound8K
from echolens.features.melspec import SR, wav_to_logmel

DATASET_ROOT = "data/urbansound8k"
FIXTURE_PATH = "../web/tests/fixtures/parity_reference.npz"

requires_dataset = pytest.mark.skipif(
    not os.path.isdir(DATASET_ROOT),
    reason="UrbanSound8K not downloaded in this environment; skipping fixture generation.",
)


@requires_dataset
def test_generate_parity_reference() -> None:
    """Generate a fixed input/output pair used by Phase 3 to cross-check the
    JavaScript feature extractor against this Python implementation."""
    ds = UrbanSound8K(folds=[10], cache_dir="data/cache")
    clip = ds.clips[0]  # any fixed clip works, as long as it's deterministic

    y, sr = clip.audio
    if sr != SR:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = y.astype(np.float32)

    logmel = wav_to_logmel(y)

    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    np.savez(FIXTURE_PATH, waveform=y, logmel=logmel)

    assert os.path.exists(FIXTURE_PATH)
