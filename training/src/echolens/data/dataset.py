import os

import numpy as np
import soundata
import torch
from torch.utils.data import Dataset

from echolens.features.melspec import wav_to_logmel, SR


class UrbanSound8K(Dataset):
    """PyTorch Dataset over a subset of UrbanSound8K folds, with cached log-mel features."""

    def __init__(self, folds: list[int], cache_dir: str, augment: bool = False):
        ds = soundata.initialize("urbansound8k", data_home="training/data/urbansound8k")
        all_clips = ds.load_clips()
        self.clips = [c for c in all_clips.values() if c.fold in folds]
        self.cache_dir = cache_dir
        self.augment = augment
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.clips)

    def _cache_path(self, clip_id: str) -> str:
        return os.path.join(self.cache_dir, f"{clip_id}.npy")

    def _load_or_compute_logmel(self, clip) -> np.ndarray:
        path = self._cache_path(clip.clip_id)
        if os.path.exists(path):
            return np.load(path)

        y, sr = clip.audio  # raw waveform + its native sample rate
        if sr != SR:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)

        logmel = wav_to_logmel(y.astype(np.float32))
        np.save(path, logmel)
        return logmel

    def __getitem__(self, i):
        clip = self.clips[i]
        logmel = self._load_or_compute_logmel(clip)

        if self.augment:
            from echolens.features.augment import spec_augment  # added in step 1.5
            logmel = spec_augment(logmel)

        x = torch.from_numpy(logmel).unsqueeze(0)  # (1, N_MELS, T)
        y = clip.class_id  # 0..9
        return x, y