"""
antispoof_dataset.py — Phase 2: PyTorch Dataset for replay anti-spoofing.

Supports two operating modes:
  1. **Single-frame mode** (default)  — Returns one 224×224 image tensor.
     Ideal for training a baseline CNN (e.g. ResNet, EfficientNet).

  2. **Sequence mode** (`seq_len > 1`) — Returns N consecutive frames from the
     same source video as a (N, C, H, W) tensor.  Ideal for temporal models
     (e.g. LSTM, 3D-CNN, Transformer).

Data source:
  Phase 1 (`extract_faces.py`) generates:
    data/frames/{dataset_name}/{real,spoof}/frame_XXXXXX.jpg
    data/frames/{dataset_name}/labels.csv
  The CSV has columns: filename, label, source, source_type, frame_idx

Label encoding:
  real  → 0
  spoof → 1

Usage:
    from antispoof_dataset import AntiSpoofDataset, get_transforms

    # ── Single-frame mode ────────────────────────────────
    ds = AntiSpoofDataset(
        csv_path="data/frames/my_dataset/labels.csv",
        root_dir="data/frames/my_dataset",
        transform=get_transforms(train=True),
    )
    img, label = ds[0]   # img shape: (3, 224, 224)

    # ── Sequence mode (8 consecutive frames) ─────────────
    ds_seq = AntiSpoofDataset(
        csv_path="data/frames/my_dataset/labels.csv",
        root_dir="data/frames/my_dataset",
        transform=get_transforms(train=True),
        seq_len=8,
    )
    seq, label = ds_seq[0]  # seq shape: (8, 3, 224, 224)
"""

# ═══════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════
import os
import csv
import random
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision.transforms as T

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# Label mapping: text label → integer class
LABEL_MAP: Dict[str, int] = {
    "real": 0,
    "spoof": 1,
}

# ImageNet mean/std used for normalisation (standard for pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1.  Transform builders
# ═══════════════════════════════════════════════════════════════
def get_transforms(
    train: bool = True,
    img_size: int = 224,
) -> T.Compose:
    """
    Build a torchvision Compose pipeline for training or evaluation.

    Training transforms include data augmentation:
      - Random horizontal flip
      - Random brightness / contrast jitter
      - Random Gaussian blur
      - ToTensor + ImageNet normalisation

    Evaluation transforms are deterministic:
      - Resize + ToTensor + ImageNet normalisation

    Parameters
    ----------
    train : bool
        If True, include augmentations; otherwise only resize + normalise.
    img_size : int
        Target spatial size (default 224).

    Returns
    -------
    torchvision.transforms.Compose
        Ready-to-use transform pipeline.
    """
    if train:
        return T.Compose([
            # ── Spatial augmentations ────────────────────────
            T.Resize((img_size, img_size)),         # ensure exact size
            T.RandomHorizontalFlip(p=0.5),          # mirror randomly

            # ── Colour / brightness augmentations ────────────
            T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),

            # ── Blur augmentation (simulates low-quality capture) ──
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),

            # ── Convert to tensor and normalise ──────────────
            T.ToTensor(),                           # HWC uint8 → CHW float [0,1]
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ═══════════════════════════════════════════════════════════════
# 2.  CSV loader — parse labels.csv from Phase 1
# ═══════════════════════════════════════════════════════════════
def load_csv(csv_path: str) -> List[Dict[str, str]]:
    """
    Read the labels CSV produced by extract_faces.py.

    Expected columns: filename, label, source, source_type, frame_idx.
    Rows with labels not in LABEL_MAP are skipped (with a warning).

    Parameters
    ----------
    csv_path : str
        Path to labels.csv.

    Returns
    -------
    List[Dict[str, str]]
        List of row dicts, each containing the CSV column values.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: List[Dict[str, str]] = []
    skipped = 0

    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = row.get("label", "").strip().lower()
            if label not in LABEL_MAP:
                skipped += 1
                continue
            rows.append(row)

    if skipped > 0:
        logger.warning("Skipped %d rows with unknown labels in %s", skipped, csv_path)

    logger.info("Loaded %d samples from %s", len(rows), csv_path)
    return rows


# ═══════════════════════════════════════════════════════════════
# 3.  Sequence grouping — group frames by source video
# ═══════════════════════════════════════════════════════════════
def group_by_source(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Group CSV rows by their 'source' field (the originating video/image).

    This is used in sequence mode so we can sample N consecutive frames
    from the same video. Frames within each group are sorted by frame_idx
    to maintain temporal order.

    Example grouping:
        { "clip01.mp4": [row0, row1, row5, row8, ...],
          "clip02.mp4": [row2, row3, ...],
          "selfie.jpg": [row4] }

    Parameters
    ----------
    rows : List[Dict[str, str]]
        All CSV rows.

    Returns
    -------
    Dict[str, List[Dict[str, str]]]
        source_name → sorted list of rows from that source.
    """
    groups: Dict[str, List[Dict[str, str]]] = {}

    for row in rows:
        source = row.get("source", "unknown")
        groups.setdefault(source, []).append(row)

    # Sort each group by frame_idx for correct temporal order
    for source in groups:
        groups[source].sort(key=lambda r: int(r.get("frame_idx", 0)))

    return groups


# ═══════════════════════════════════════════════════════════════
# 4.  Main Dataset class
# ═══════════════════════════════════════════════════════════════
class AntiSpoofDataset(Dataset):
    """
    PyTorch Dataset for face anti-spoofing (replay detection).

    Operates in two modes controlled by `seq_len`:

    ┌──────────────────────────────────────────────────────────┐
    │  SINGLE-FRAME mode  (seq_len=1, default)                │
    │  ─────────────────────────────────────                   │
    │  Each sample = one image.                                │
    │  __getitem__ returns: (tensor[3,H,W], label_int)         │
    │  Best for: CNN baselines (ResNet, EfficientNet, etc.)    │
    └──────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────┐
    │  SEQUENCE mode  (seq_len > 1)                            │
    │  ───────────────                                         │
    │  Each sample = N consecutive frames from one source video│
    │  __getitem__ returns: (tensor[N,3,H,W], label_int)       │
    │  Best for: temporal models (LSTM, 3D-CNN, Transformer)   │
    │                                                          │
    │  SEQUENCE SAMPLING STRATEGY:                             │
    │  1. Group all frames by source video.                    │
    │  2. For each group with >= seq_len frames, create one    │
    │     or more contiguous windows of length seq_len.        │
    │  3. Each window becomes one dataset sample.              │
    │  4. If a group has fewer than seq_len frames, it is      │
    │     padded by repeating the last frame (edge-padding)    │
    │     so no data is thrown away.                            │
    │  5. At training time windows are randomly offset for     │
    │     diversity; at eval time they start at index 0.       │
    └──────────────────────────────────────────────────────────┘

    Parameters
    ----------
    csv_path : str
        Path to the labels.csv produced by Phase 1.
    root_dir : str
        Directory containing real/ and spoof/ subfolders with images.
    transform : callable, optional
        torchvision-style transform applied to each PIL image.
    seq_len : int
        Number of consecutive frames per sample (1 = single-frame mode).
    stride : int
        Window stride for generating sequence samples from long videos.
        Only used when seq_len > 1. Smaller stride = more overlap = more
        training samples.
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        transform: Optional[Callable] = None,
        seq_len: int = 1,
        stride: int = 1,
    ):
        # ── Store parameters ─────────────────────────────────
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = max(1, seq_len)
        self.stride = max(1, stride)

        # ── Load and validate CSV rows ───────────────────────
        self.all_rows = load_csv(csv_path)
        if len(self.all_rows) == 0:
            raise ValueError(f"No valid samples found in {csv_path}")

        # ── Build the index (list of samples) ────────────────
        if self.seq_len == 1:
            # SINGLE-FRAME MODE
            # Each CSV row is one sample. The index is trivially the row list.
            self.samples = self.all_rows
        else:
            # SEQUENCE MODE
            # Group frames by source, then create sliding windows.
            self.samples = self._build_sequence_index()

        logger.info(
            "AntiSpoofDataset ready: %d samples  (seq_len=%d, mode=%s)",
            len(self.samples),
            self.seq_len,
            "sequence" if self.seq_len > 1 else "single-frame",
        )

    # ──────────────────────────────────────────────────────────
    # Sequence index builder
    # ──────────────────────────────────────────────────────────
    def _build_sequence_index(self) -> List[List[Dict[str, str]]]:
        """
        Create a list of sequence windows for temporal mode.

        Algorithm:
          1. Group rows by source video/image.
          2. For each group:
             a. If len(group) >= seq_len:
                Slide a window of size seq_len with step = self.stride
                across the temporally-sorted frames.
             b. If len(group) < seq_len:
                Pad by repeating the last frame until we reach seq_len.
                This yields exactly one sample from that group.
          3. Return a flat list of windows. Each window is a list of
             seq_len row dicts.

        Returns
        -------
        List[List[Dict[str, str]]]
            Each element is a list of seq_len row dicts forming one sample.
        """
        groups = group_by_source(self.all_rows)
        windows: List[List[Dict[str, str]]] = []

        for source, frames in groups.items():
            n = len(frames)

            if n >= self.seq_len:
                # ── Sliding window ───────────────────────────
                # Start positions: 0, stride, 2*stride, ...
                # Last valid start: n - seq_len
                for start in range(0, n - self.seq_len + 1, self.stride):
                    window = frames[start : start + self.seq_len]
                    windows.append(window)
            else:
                # ── Edge-padding: repeat last frame ──────────
                # E.g. if seq_len=8 but source has 3 frames:
                # [f0, f1, f2, f2, f2, f2, f2, f2]
                padded = frames + [frames[-1]] * (self.seq_len - n)
                windows.append(padded)

        logger.info(
            "Built %d sequence windows from %d source groups (stride=%d)",
            len(windows), len(groups), self.stride,
        )
        return windows

    # ──────────────────────────────────────────────────────────
    # __len__
    # ──────────────────────────────────────────────────────────
    def __len__(self) -> int:
        """Return total number of samples (frames or sequences)."""
        return len(self.samples)

    # ──────────────────────────────────────────────────────────
    # __getitem__
    # ──────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Fetch one sample by index.

        Single-frame mode
        -----------------
        Returns (tensor[3, H, W], label_int).

        Sequence mode
        -------------
        Returns (tensor[seq_len, 3, H, W], label_int).

        How frames / images are paired with labels:
          - Each CSV row already contains a 'label' field ("real" or "spoof")
            assigned during Phase 1 based on the source file's path.
          - In single-frame mode: the label comes directly from that row.
          - In sequence mode: all frames in a window share the same source
            video, so they share the same label. We take the label from the
            first frame in the window.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Tuple[torch.Tensor, int]
            (image_or_sequence_tensor, label_int)
        """
        if self.seq_len == 1:
            # ── SINGLE-FRAME: load one image ─────────────────
            row = self.samples[idx]
            image = self._load_image(row)
            label = LABEL_MAP[row["label"].strip().lower()]
            return image, label
        else:
            # ── SEQUENCE: load N consecutive frames ──────────
            window = self.samples[idx]
            frames = [self._load_image(row) for row in window]

            # Stack into (seq_len, C, H, W) tensor
            sequence = torch.stack(frames, dim=0)

            # Label from the first frame (all frames in window share source)
            label = LABEL_MAP[window[0]["label"].strip().lower()]
            return sequence, label

    # ──────────────────────────────────────────────────────────
    # Image loader
    # ──────────────────────────────────────────────────────────
    def _load_image(self, row: Dict[str, str]) -> torch.Tensor:
        """
        Load a single image from disk and apply transforms.

        The file path is constructed from root_dir + label subfolder + filename.
        For example: data/frames/my_dataset/real/frame_000042.jpg

        Parameters
        ----------
        row : Dict[str, str]
            CSV row dict with keys: filename, label, source, source_type, frame_idx.

        Returns
        -------
        torch.Tensor
            Transformed image tensor of shape (C, H, W).
        """
        label = row["label"].strip().lower()
        filename = row["filename"]

        # Build full path: root_dir / {real|spoof} / filename
        img_path = os.path.join(self.root_dir, label, filename)

        # Load as RGB PIL Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to load %s: %s — returning black image", img_path, exc)
            # Return a black placeholder so training doesn't crash on one bad file
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Apply transforms (augmentation + ToTensor + normalisation)
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Fallback: at minimum convert to tensor
            image = T.ToTensor()(image)

        return image

    # ──────────────────────────────────────────────────────────
    # Utility: class distribution
    # ──────────────────────────────────────────────────────────
    def get_label_counts(self) -> Dict[str, int]:
        """
        Count samples per class. Useful for computing class weights.

        Returns
        -------
        Dict[str, int]
            {"real": N, "spoof": M}
        """
        counts = {"real": 0, "spoof": 0}

        if self.seq_len == 1:
            for row in self.samples:
                lbl = row["label"].strip().lower()
                counts[lbl] = counts.get(lbl, 0) + 1
        else:
            for window in self.samples:
                lbl = window[0]["label"].strip().lower()
                counts[lbl] = counts.get(lbl, 0) + 1

        return counts

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for imbalanced datasets.

        Returns
        -------
        torch.Tensor
            Tensor of shape (2,) with [weight_real, weight_spoof].
        """
        counts = self.get_label_counts()
        total = counts["real"] + counts["spoof"]
        if total == 0:
            return torch.tensor([1.0, 1.0])

        # Inverse frequency: weight = total / (num_classes * count)
        w_real  = total / (2.0 * max(counts["real"], 1))
        w_spoof = total / (2.0 * max(counts["spoof"], 1))
        return torch.tensor([w_real, w_spoof], dtype=torch.float32)
