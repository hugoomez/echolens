import os

import pytest

from echolens.data.dataset import UrbanSound8K

CACHE_DIR = "data/cache"
DATASET_ROOT = "data/urbansound8k"

requires_dataset = pytest.mark.skipif(
    not os.path.isdir(DATASET_ROOT),
    reason="UrbanSound8K not downloaded in this environment; skipping data-dependent test.",
)


@requires_dataset
def test_fold_splits_do_not_overlap() -> None:
    """Clips assigned to different folds must never overlap."""
    train_ds = UrbanSound8K(folds=list(range(1, 10)), cache_dir=CACHE_DIR)
    val_ds = UrbanSound8K(folds=[10], cache_dir=CACHE_DIR)

    train_ids = {c.clip_id for c in train_ds.clips}
    val_ids = {c.clip_id for c in val_ds.clips}

    assert train_ids.isdisjoint(val_ids)


@requires_dataset
def test_labels_are_in_valid_range() -> None:
    """All class labels must fall within 0..9 (the 10 UrbanSound8K classes)."""
    ds = UrbanSound8K(folds=[10], cache_dir=CACHE_DIR)
    for i in range(len(ds)):
        _, label = ds[i]
        assert 0 <= label <= 9
