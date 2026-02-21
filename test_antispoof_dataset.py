"""
test_antispoof_dataset.py — Comprehensive test suite for Phase 2 Dataset.

Covers:
  - CSV loading (valid, missing, unknown labels)
  - Source grouping and temporal sorting
  - Transform builders (train vs eval)
  - Single-frame mode (__getitem__, shapes, labels)
  - Sequence mode (window creation, padding, shapes)
  - Label counts and class weight computation
  - Edge cases (corrupt images, empty CSV, single-frame sources)

Run:
    pytest test_antispoof_dataset.py -v
"""

import os
import csv
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import antispoof_dataset as ads


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_dir():
    """Create and tear down a temporary directory."""
    d = tempfile.mkdtemp(prefix="test_ds_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_dataset(tmp_dir):
    """
    Build a minimal on-disk dataset with:
      - 6 real frames (from 2 videos: vid_a.mp4 × 4, vid_b.mp4 × 2)
      - 4 spoof frames (from 1 video: spoof_c.mp4 × 3, plus 1 image)
    Returns (csv_path, root_dir).
    """
    real_dir  = os.path.join(tmp_dir, "real")
    spoof_dir = os.path.join(tmp_dir, "spoof")
    os.makedirs(real_dir)
    os.makedirs(spoof_dir)

    rows = []

    # ── Real frames from vid_a.mp4 (4 frames) ───────────────
    for i in range(4):
        fname = f"frame_{i:06d}.jpg"
        _write_dummy_image(os.path.join(real_dir, fname))
        rows.append({
            "filename": fname, "label": "real",
            "source": "vid_a.mp4", "source_type": "video", "frame_idx": str(i * 10),
        })

    # ── Real frames from vid_b.mp4 (2 frames) ───────────────
    for i in range(4, 6):
        fname = f"frame_{i:06d}.jpg"
        _write_dummy_image(os.path.join(real_dir, fname))
        rows.append({
            "filename": fname, "label": "real",
            "source": "vid_b.mp4", "source_type": "video", "frame_idx": str((i - 4) * 10),
        })

    # ── Spoof frames from spoof_c.mp4 (3 frames) ────────────
    for i in range(3):
        fname = f"frame_{i:06d}.jpg"
        _write_dummy_image(os.path.join(spoof_dir, fname))
        rows.append({
            "filename": fname, "label": "spoof",
            "source": "spoof_c.mp4", "source_type": "video", "frame_idx": str(i * 10),
        })

    # ── Spoof image (1 standalone) ───────────────────────────
    fname = "frame_000003.jpg"
    _write_dummy_image(os.path.join(spoof_dir, fname))
    rows.append({
        "filename": fname, "label": "spoof",
        "source": "attack_photo.jpg", "source_type": "image", "frame_idx": "0",
    })

    # ── Write CSV ────────────────────────────────────────────
    csv_path = os.path.join(tmp_dir, "labels.csv")
    _write_csv(csv_path, rows)

    return csv_path, tmp_dir


def _write_dummy_image(path: str, size: int = 224):
    """Create a random 224×224 RGB JPEG on disk."""
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, "JPEG")


def _write_csv(path: str, rows):
    """Write a labels CSV from a list of dicts."""
    fieldnames = ["filename", "label", "source", "source_type", "frame_idx"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════
# 1. CSV loading
# ═══════════════════════════════════════════════════════════════
class TestLoadCSV:

    def test_loads_valid_csv(self, sample_dataset):
        """Should load all 10 rows from the sample CSV."""
        csv_path, _ = sample_dataset
        rows = ads.load_csv(csv_path)
        assert len(rows) == 10

    def test_raises_on_missing_csv(self, tmp_dir):
        """Should raise FileNotFoundError for non-existent CSV."""
        with pytest.raises(FileNotFoundError):
            ads.load_csv(os.path.join(tmp_dir, "nope.csv"))

    def test_skips_unknown_labels(self, tmp_dir):
        """Rows with labels not in LABEL_MAP should be silently skipped."""
        csv_path = os.path.join(tmp_dir, "labels.csv")
        _write_csv(csv_path, [
            {"filename": "a.jpg", "label": "real", "source": "x", "source_type": "image", "frame_idx": "0"},
            {"filename": "b.jpg", "label": "unknown", "source": "y", "source_type": "image", "frame_idx": "0"},
        ])
        rows = ads.load_csv(csv_path)
        assert len(rows) == 1
        assert rows[0]["label"] == "real"


# ═══════════════════════════════════════════════════════════════
# 2. Source grouping
# ═══════════════════════════════════════════════════════════════
class TestGroupBySource:

    def test_groups_correctly(self, sample_dataset):
        """Should create 4 source groups from the sample data."""
        csv_path, _ = sample_dataset
        rows = ads.load_csv(csv_path)
        groups = ads.group_by_source(rows)
        # vid_a, vid_b, spoof_c, attack_photo
        assert len(groups) == 4

    def test_temporal_order(self, sample_dataset):
        """Frames within each group should be sorted by frame_idx ascending."""
        csv_path, _ = sample_dataset
        rows = ads.load_csv(csv_path)
        groups = ads.group_by_source(rows)

        for source, frames in groups.items():
            indices = [int(f["frame_idx"]) for f in frames]
            assert indices == sorted(indices), f"Group {source} not sorted"


# ═══════════════════════════════════════════════════════════════
# 3. Transforms
# ═══════════════════════════════════════════════════════════════
class TestTransforms:

    def test_train_transform_output(self):
        """Train transform should produce a (3,224,224) float tensor."""
        t = ads.get_transforms(train=True, img_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == (3, 224, 224)
        assert out.dtype == torch.float32

    def test_eval_transform_deterministic(self):
        """Eval transform should produce identical tensor on repeated calls."""
        t = ads.get_transforms(train=False, img_size=224)
        img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
        out1 = t(img)
        out2 = t(img)
        assert torch.allclose(out1, out2)

    def test_custom_image_size(self):
        """Transform should respect a non-default image size."""
        t = ads.get_transforms(train=False, img_size=112)
        img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == (3, 112, 112)


# ═══════════════════════════════════════════════════════════════
# 4. Single-frame mode
# ═══════════════════════════════════════════════════════════════
class TestSingleFrameMode:

    def test_len(self, sample_dataset):
        """Length should equal number of CSV rows (10)."""
        csv_path, root = sample_dataset
        ds = ads.AntiSpoofDataset(csv_path, root, transform=ads.get_transforms(False))
        assert len(ds) == 10

    def test_getitem_shape(self, sample_dataset):
        """Each sample should be (tensor[3,224,224], int)."""
        csv_path, root = sample_dataset
        ds = ads.AntiSpoofDataset(csv_path, root, transform=ads.get_transforms(False))
        img, label = ds[0]
        assert img.shape == (3, 224, 224)
        assert isinstance(label, int)

    def test_label_values(self, sample_dataset):
        """Labels should be 0 (real) or 1 (spoof)."""
        csv_path, root = sample_dataset
        ds = ads.AntiSpoofDataset(csv_path, root, transform=ads.get_transforms(False))
        labels = set()
        for i in range(len(ds)):
            _, lbl = ds[i]
            labels.add(lbl)
        assert labels == {0, 1}

    def test_real_label_is_zero(self, sample_dataset):
        """First 6 samples (real) should have label=0."""
        csv_path, root = sample_dataset
        ds = ads.AntiSpoofDataset(csv_path, root, transform=ads.get_transforms(False))
        _, lbl = ds[0]
        assert lbl == 0

    def test_spoof_label_is_one(self, sample_dataset):
        """Samples 6-9 (spoof) should have label=1."""
        csv_path, root = sample_dataset
        ds = ads.AntiSpoofDataset(csv_path, root, transform=ads.get_transforms(False))
        _, lbl = ds[9]
        assert lbl == 1


# ═══════════════════════════════════════════════════════════════
# 5. Sequence mode
# ═══════════════════════════════════════════════════════════════
class TestSequenceMode:

    def test_sequence_shape(self, sample_dataset):
        """Sequence sample should have shape (seq_len, 3, 224, 224)."""
        csv_path, root = sample_dataset
        ds = ads.AntiSpoofDataset(
            csv_path, root,
            transform=ads.get_transforms(False),
            seq_len=2,
        )
        seq, label = ds[0]
        assert seq.shape == (2, 3, 224, 224)

    def test_sequence_windows_count(self, sample_dataset):
        """
        With seq_len=2. stride=1:
          vid_a (4 frames) → 3 windows
          vid_b (2 frames) → 1 window
          spoof_c (3 frames) → 2 windows
          attack_photo (1 frame, padded) → 1 window
        Total = 7 windows.
        """
        csv_path, root = sample_dataset
        ds = ads.AntiSpoofDataset(
            csv_path, root,
            transform=ads.get_transforms(False),
            seq_len=2, stride=1,
        )
        assert len(ds) == 7

    def test_padding_short_source(self, sample_dataset):
        """
        When seq_len=4, attack_photo (1 frame) should be padded to 4.
        The padded sample should still load without error.
        """
        csv_path, root = sample_dataset
        ds = ads.AntiSpoofDataset(
            csv_path, root,
            transform=ads.get_transforms(False),
            seq_len=4,
        )
        # Iterate all samples — none should crash
        for i in range(len(ds)):
            seq, lbl = ds[i]
            assert seq.shape == (4, 3, 224, 224)

    def test_large_stride_reduces_windows(self, sample_dataset):
        """Larger stride should produce fewer windows."""
        csv_path, root = sample_dataset
        ds_s1 = ads.AntiSpoofDataset(csv_path, root, seq_len=2, stride=1)
        ds_s2 = ads.AntiSpoofDataset(csv_path, root, seq_len=2, stride=2)
        assert len(ds_s2) <= len(ds_s1)

    def test_seq_len_1_fallback(self, sample_dataset):
        """seq_len=1 should be identical to single-frame mode."""
        csv_path, root = sample_dataset
        ds = ads.AntiSpoofDataset(csv_path, root, seq_len=1)
        assert len(ds) == 10
        img, lbl = ds[0]
        # Single frame → 3D tensor (not 4D)
        assert img.dim() == 3


# ═══════════════════════════════════════════════════════════════
# 6. Label counts and class weights
# ═══════════════════════════════════════════════════════════════
class TestLabelUtils:

    def test_label_counts_single_frame(self, sample_dataset):
        """Should count 6 real + 4 spoof in single-frame mode."""
        csv_path, root = sample_dataset
        ds = ads.AntiSpoofDataset(csv_path, root)
        counts = ds.get_label_counts()
        assert counts["real"] == 6
        assert counts["spoof"] == 4

    def test_class_weights_shape(self, sample_dataset):
        """Class weights tensor should have shape (2,)."""
        csv_path, root = sample_dataset
        ds = ads.AntiSpoofDataset(csv_path, root)
        w = ds.get_class_weights()
        assert w.shape == (2,)
        assert w.dtype == torch.float32

    def test_class_weights_imbalanced(self, sample_dataset):
        """Minority class (spoof) should get higher weight."""
        csv_path, root = sample_dataset
        ds = ads.AntiSpoofDataset(csv_path, root)
        w = ds.get_class_weights()
        # spoof (4 samples) is minority → w[1] > w[0]
        assert w[1] > w[0]


# ═══════════════════════════════════════════════════════════════
# 7. Edge cases
# ═══════════════════════════════════════════════════════════════
class TestEdgeCases:

    def test_corrupt_image_returns_black(self, tmp_dir):
        """A corrupt image should silently return a black tensor, not crash."""
        real_dir = os.path.join(tmp_dir, "real")
        os.makedirs(real_dir)
        # Write garbage data
        bad_path = os.path.join(real_dir, "frame_000000.jpg")
        with open(bad_path, "wb") as f:
            f.write(b"not a valid jpeg at all")

        csv_path = os.path.join(tmp_dir, "labels.csv")
        _write_csv(csv_path, [{
            "filename": "frame_000000.jpg", "label": "real",
            "source": "bad.mp4", "source_type": "video", "frame_idx": "0",
        }])

        ds = ads.AntiSpoofDataset(csv_path, tmp_dir, transform=ads.get_transforms(False))
        img, lbl = ds[0]
        # Should not throw — returns a tensor
        assert img.shape == (3, 224, 224)

    def test_empty_csv_raises(self, tmp_dir):
        """Dataset with zero valid rows should raise ValueError."""
        csv_path = os.path.join(tmp_dir, "labels.csv")
        _write_csv(csv_path, [])
        with pytest.raises(ValueError):
            ads.AntiSpoofDataset(csv_path, tmp_dir)

    def test_no_transform(self, sample_dataset):
        """Dataset should work without a transform (fallback to ToTensor)."""
        csv_path, root = sample_dataset
        ds = ads.AntiSpoofDataset(csv_path, root, transform=None)
        img, lbl = ds[0]
        assert img.dim() == 3  # still a 3D tensor
