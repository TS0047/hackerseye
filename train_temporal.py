"""
train_temporal.py — Phase 4: Temporal anti-spoofing model (CNN + temporal aggregator).

Uses a pretrained CNN backbone to extract per-frame features, then aggregates
them across time using either **temporal pooling** or an **LSTM** before a
binary classifier head.

Architecture overview:
    Input: (B, T, C, H, W) — batch of frame sequences
                                ↓
    CNN backbone (ResNet-18 / MobileNetV3) — extracts per-frame features
    Output: (B, T, D) — D-dimensional feature per frame
                                ↓
    Temporal aggregator — collapses T frames into one descriptor
      Option A: Temporal pooling (avg or max across T)  → (B, D)
      Option B: LSTM over T timesteps, take last hidden → (B, D)
                                ↓
    Classifier head — Linear(D, 1) → binary logit
                                ↓
    BCEWithLogitsLoss (with pos_weight for class imbalance)

Why temporal matters:
    Replay / screen attacks often exhibit subtle temporal artefacts that
    single-frame models miss — screen flicker, moiré patterns, unnatural
    motion. Aggregating across frames lets the model exploit these cues.

Usage:
    # Train with LSTM aggregator (default)
    python train_temporal.py --csv data/frames/my_dataset/labels.csv \\
                             --root data/frames/my_dataset \\
                             --seq-len 8 --aggregator lstm

    # Train with temporal average pooling
    python train_temporal.py --csv data/frames/my_dataset/labels.csv \\
                             --root data/frames/my_dataset \\
                             --seq-len 8 --aggregator avg

    # Full control — MobileNetV3 backbone, frozen, with scheduler
    python train_temporal.py --csv data/frames/my_dataset/labels.csv \\
                             --root data/frames/my_dataset \\
                             --backbone mobilenet --freeze-backbone \\
                             --seq-len 16 --aggregator lstm \\
                             --epochs 30 --batch-size 8 --lr 3e-4 --scheduler
"""

# ═══════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════
import argparse
import os
import sys
import time
import logging
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import torchvision.models as models

# ── Metrics (reuse Phase 3 helpers) ──────────────────────────
from train_antispoof import (
    compute_metrics,
    print_confusion_matrix,
    save_checkpoint,
    load_checkpoint,
)

# ── Phase 2 dataset + transforms ────────────────────────────
from antispoof_dataset import (
    AntiSpoofDataset,
    get_transforms,
    LABEL_MAP,
)

# ═══════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. CNN Backbone — feature extractor (no classifier head)
# ═══════════════════════════════════════════════════════════════

def _build_backbone(
    name: str = "resnet18",
    pretrained: bool = True,
    freeze: bool = False,
) -> Tuple[nn.Module, int]:
    """
    Build a CNN backbone and strip its classifier head.

    Parameters
    ----------
    name : str
        "resnet18" or "mobilenet" (MobileNetV3-Small).
    pretrained : bool
        Use ImageNet weights.
    freeze : bool
        Freeze all backbone parameters (only temporal + head will train).

    Returns
    -------
    Tuple[nn.Module, int]
        (backbone_module, feature_dim)  — feature_dim is the output vector
        size after global average pooling.
    """
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        feat_dim = base.fc.in_features  # 512

        # Remove the final FC layer — keep everything up to avgpool
        # nn.Sequential of all layers except fc
        backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,       # output: (B, 512, 1, 1)
            nn.Flatten(1),      # output: (B, 512)
        )
    elif name == "mobilenet":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)
        feat_dim = base.classifier[0].in_features  # 576

        # MobileNetV3: features → avgpool → classifier
        # Keep features + avgpool, drop classifier
        backbone = nn.Sequential(
            base.features,
            base.avgpool,       # output: (B, 576, 1, 1)
            nn.Flatten(1),      # output: (B, 576)
        )
    else:
        raise ValueError(f"Unknown backbone: {name!r}. Use 'resnet18' or 'mobilenet'.")

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone (%s) frozen — %d params locked.", name, sum(p.numel() for p in backbone.parameters()))

    logger.info("Backbone: %s  →  feature dim = %d", name, feat_dim)
    return backbone, feat_dim


# ═══════════════════════════════════════════════════════════════
# 2. Temporal aggregators
# ═══════════════════════════════════════════════════════════════

class TemporalAvgPool(nn.Module):
    """
    Average-pool across the time dimension.

    Input:  (B, T, D)
    Output: (B, D)

    Simplest aggregator — treats all frames equally, captures the "average
    appearance" across the sequence.  Fast and parameter-free.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)   # average over T


class TemporalMaxPool(nn.Module):
    """
    Max-pool across the time dimension.

    Input:  (B, T, D)
    Output: (B, D)

    Picks the strongest activation per feature across frames — good for
    detecting intermittent artefacts (e.g. a single-frame flicker).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=1).values   # max over T


class TemporalLSTM(nn.Module):
    """
    Bidirectional LSTM over the frame-feature sequence.

    Input:  (B, T, D)
    Output: (B, hidden_size * 2)  — concatenated final hidden states
            from forward and backward passes

    Why LSTM?
      An LSTM can learn *order-dependent* patterns — e.g. periodic screen
      refresh causing brightness oscillation every N frames.  Pooling methods
      discard order; the LSTM explicitly models it.

    How sequences are processed:
      1. Each frame's D-dim feature vector is one timestep input.
      2. The LSTM reads the sequence left-to-right (and right-to-left since
         bidirectional=True).
      3. We take the final hidden state from both directions, concatenate
         them, and pass through a small projection layer.

    Parameters
    ----------
    input_dim : int
        Feature dimension from the CNN backbone (e.g. 512 for ResNet-18).
    hidden_size : int
        LSTM hidden state size. Output will be hidden_size * 2 (bidirectional).
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers (only used when num_layers > 1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # expect (B, T, D) input
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Output dimension after concat of forward + backward hidden
        self.output_dim = hidden_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, T, D)

        Returns
        -------
        Tensor of shape (B, hidden_size * 2)
        """
        # lstm_out: (B, T, hidden*2), (h_n, c_n)
        # h_n: (num_layers*2, B, hidden) — last hidden for each layer/direction
        _, (h_n, _) = self.lstm(x)

        # Take the final layer's forward and backward hidden states
        # h_n shape: (num_layers * num_directions, B, hidden_size)
        # Forward final: h_n[-2], Backward final: h_n[-1]
        h_forward  = h_n[-2]   # (B, hidden_size)
        h_backward = h_n[-1]   # (B, hidden_size)

        # Concatenate: (B, hidden_size * 2)
        return torch.cat([h_forward, h_backward], dim=1)


# ═══════════════════════════════════════════════════════════════
# 3. Full temporal model — backbone + aggregator + head
# ═══════════════════════════════════════════════════════════════

class TemporalAntiSpoofModel(nn.Module):
    """
    Temporal anti-spoofing classifier.

    Combines:
      1. CNN backbone  — extracts per-frame features    (B*T, D)
      2. Temporal aggregator — collapses frames          (B, D') where D'=D or hidden*2
      3. Classifier head — binary logit                  (B, 1)

    How sequences are batched and processed:
      - Input tensor has shape (B, T, C, H, W):
          B = batch size (number of video clips)
          T = sequence length (frames per clip)
          C, H, W = 3, 224, 224  (RGB image)
      - We reshape to (B*T, C, H, W) to process all frames through the CNN
        backbone in one efficient forward pass.
      - Reshape back to (B, T, D) for the temporal aggregator.
      - Aggregator reduces T → 1, giving (B, D').
      - Classifier head maps D' → 1 logit.

    Parameters
    ----------
    backbone_name : str
        "resnet18" or "mobilenet".
    aggregator : str
        "avg", "max", or "lstm".
    pretrained : bool
        Use ImageNet backbone weights.
    freeze_backbone : bool
        Freeze CNN backbone parameters.
    lstm_hidden : int
        LSTM hidden size (only used when aggregator="lstm").
    lstm_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate in classifier head (and LSTM if layers > 1).
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        aggregator: str = "lstm",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── 1. CNN backbone ──────────────────────────────────
        self.backbone, feat_dim = _build_backbone(
            name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )

        # ── 2. Temporal aggregator ───────────────────────────
        if aggregator == "avg":
            self.temporal = TemporalAvgPool()
            agg_out_dim = feat_dim
        elif aggregator == "max":
            self.temporal = TemporalMaxPool()
            agg_out_dim = feat_dim
        elif aggregator == "lstm":
            self.temporal = TemporalLSTM(
                input_dim=feat_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                dropout=dropout,
            )
            agg_out_dim = self.temporal.output_dim  # hidden * 2
        else:
            raise ValueError(
                f"Unknown aggregator: {aggregator!r}. "
                "Choose from 'avg', 'max', 'lstm'."
            )

        # ── 3. Classifier head ───────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(agg_out_dim, 1),   # single logit for BCEWithLogitsLoss
        )

        self._aggregator_name = aggregator
        self._feat_dim = feat_dim

        total_params = sum(p.numel() for p in self.parameters())
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "TemporalAntiSpoofModel: backbone=%s  aggregator=%s  "
            "feat_dim=%d  agg_out=%d  total_params=%s  trainable=%s",
            backbone_name, aggregator, feat_dim, agg_out_dim,
            f"{total_params:,}", f"{train_params:,}",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of frame sequences.

        Parameters
        ----------
        x : Tensor of shape (B, T, C, H, W)
            Batch of frame sequences.

        Returns
        -------
        Tensor of shape (B, 1)
            Raw logits (pass through sigmoid for probabilities).
        """
        B, T, C, H, W = x.shape

        # ── Step 1: Flatten batch and time → (B*T, C, H, W) ─
        # This lets us run all frames through the CNN in one shot.
        frames = x.view(B * T, C, H, W)

        # ── Step 2: Extract per-frame features → (B*T, D) ───
        features = self.backbone(frames)

        # ── Step 3: Reshape back to (B, T, D) ───────────────
        features = features.view(B, T, -1)

        # ── Step 4: Temporal aggregation → (B, D') ──────────
        aggregated = self.temporal(features)

        # ── Step 5: Classify → (B, 1) ───────────────────────
        logits = self.classifier(aggregated)

        return logits


# ═══════════════════════════════════════════════════════════════
# 4. Training loop — one epoch (handles sequence batches)
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Train the temporal model for one epoch.

    How loss is computed:
      - The dataloader yields (sequences, labels) where:
          sequences: (B, T, C, H, W)  — B clips of T frames each
          labels:    (B,)              — one label per clip (real=0, spoof=1)
      - Model outputs (B, 1) logits, squeezed to (B,).
      - BCEWithLogitsLoss computes binary cross-entropy on the per-clip
        prediction vs. the per-clip label.
      - Class imbalance is handled by pos_weight in the loss function.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        (average_loss, metrics_dict)
    """
    model.train()
    running_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []

    for sequences, labels in loader:
        # sequences: (B, T, C, H, W)   labels: (B,) int
        sequences = sequences.to(device)
        labels_f  = labels.float().to(device)

        # ── Forward ──────────────────────────────────────────
        logits = model(sequences).squeeze(1)  # (B,)
        loss = criterion(logits, labels_f)

        # ── Backward ─────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Track stats ──────────────────────────────────────
        running_loss += loss.item() * sequences.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        all_labels.extend(labels.long().cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / max(len(all_labels), 1)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


# ═══════════════════════════════════════════════════════════════
# 5. Validation loop
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], List[int], List[int]]:
    """
    Evaluate temporal model on validation data.

    Returns
    -------
    Tuple[float, Dict, List[int], List[int]]
        (avg_loss, metrics, all_true_labels, all_pred_labels)
    """
    model.eval()
    running_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels_f  = labels.float().to(device)

        logits = model(sequences).squeeze(1)
        loss = criterion(logits, labels_f)

        running_loss += loss.item() * sequences.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        all_labels.extend(labels.long().cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / max(len(all_labels), 1)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_labels, all_preds


# ═══════════════════════════════════════════════════════════════
# 6. CLI argument parser
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 4 — Train temporal anti-spoof model (CNN + aggregator).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ─────────────────────────────────────────────────
    p.add_argument("--csv",  required=True, help="Path to labels.csv from Phase 1.")
    p.add_argument("--root", required=True, help="Root dir containing face crops.")

    # ── Sequence options ─────────────────────────────────────
    p.add_argument("--seq-len", type=int, default=8,
                   help="Number of consecutive frames per sample.")
    p.add_argument("--stride",  type=int, default=4,
                   help="Sliding window stride for sequence sampling.")

    # ── Model architecture ───────────────────────────────────
    p.add_argument("--backbone", choices=["resnet18", "mobilenet"], default="resnet18",
                   help="CNN backbone for feature extraction.")
    p.add_argument("--aggregator", choices=["avg", "max", "lstm"], default="lstm",
                   help="Temporal aggregation method.")
    p.add_argument("--lstm-hidden", type=int, default=128,
                   help="LSTM hidden size (only for --aggregator lstm).")
    p.add_argument("--lstm-layers", type=int, default=1,
                   help="Number of LSTM layers.")

    # ── Training hyper-params ────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch-size",   type=int,   default=8,
                   help="Batch size (sequences are memory-heavy; keep small).")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split",    type=float, default=0.2)
    p.add_argument("--dropout",      type=float, default=0.3)

    # ── Model options ────────────────────────────────────────
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Freeze CNN backbone; only train aggregator + head.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="No ImageNet weights on the backbone.")

    # ── Scheduler / checkpoint ───────────────────────────────
    p.add_argument("--scheduler", action="store_true",
                   help="Enable cosine-annealing LR scheduler.")
    p.add_argument("--checkpoint", default="checkpoints/best_temporal.pt",
                   help="Path to save best model checkpoint.")
    p.add_argument("--resume", default=None,
                   help="Resume training from a checkpoint.")

    # ── Misc ────────────────────────────────────────────────
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--workers", type=int, default=0)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# 7. Main entry point
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Reproducibility ──────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device ───────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.warning(
            "CUDA not available — training on CPU. "
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        )

    # ── Build sequence dataset ───────────────────────────────
    # Using Phase 2 AntiSpoofDataset in SEQUENCE mode (seq_len > 1).
    # Each sample = (T, C, H, W) tensor + label.
    logger.info("Loading CSV: %s  (seq_len=%d, stride=%d)", args.csv, args.seq_len, args.stride)

    full_dataset = AntiSpoofDataset(
        csv_path=args.csv,
        root_dir=args.root,
        transform=None,   # set per-split below
        seq_len=args.seq_len,
        stride=args.stride,
    )
    total = len(full_dataset)
    if total == 0:
        logger.error("Dataset is empty. Run extract_faces.py first.")
        sys.exit(1)

    # ── Train / val split ────────────────────────────────────
    val_size   = max(1, int(total * args.val_split))
    train_size = total - val_size

    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # ── Wrap with per-split transforms ───────────────────────
    # Frame-level augmentation is applied inside AntiSpoofDataset._load_image
    # (each frame in the sequence gets the same transform pipeline).
    class _SubsetWithTransform(torch.utils.data.Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    train_loader_ds = _SubsetWithTransform(
        AntiSpoofDataset(
            args.csv, args.root,
            transform=get_transforms(train=True),
            seq_len=args.seq_len, stride=args.stride,
        ),
        train_ds.indices,
    )
    val_loader_ds = _SubsetWithTransform(
        AntiSpoofDataset(
            args.csv, args.root,
            transform=get_transforms(train=False),
            seq_len=args.seq_len, stride=args.stride,
        ),
        val_ds.indices,
    )

    logger.info("Split: %d train / %d val  (%.0f%% val)", train_size, val_size, args.val_split * 100)

    # ── Label distribution ───────────────────────────────────
    counts = full_dataset.get_label_counts()
    logger.info("Label counts — real: %d, spoof: %d", counts.get("real", 0), counts.get("spoof", 0))

    # ── DataLoaders ──────────────────────────────────────────
    # NOTE: batch_size should be small for sequences to fit in GPU memory.
    # Memory per batch ≈ B × T × 3 × 224 × 224 × 4 bytes
    # E.g. B=8, T=8: ~8 × 8 × 3 × 224 × 224 × 4 ≈ 308 MB
    train_loader = DataLoader(
        train_loader_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_loader_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Build model ──────────────────────────────────────────
    model = TemporalAntiSpoofModel(
        backbone_name=args.backbone,
        aggregator=args.aggregator,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)

    # ── Loss with class-weight ───────────────────────────────
    n_real  = counts.get("real", 1)
    n_spoof = counts.get("spoof", 1)
    pos_weight = torch.tensor([n_real / max(n_spoof, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info("BCEWithLogitsLoss  pos_weight=%.3f", pos_weight.item())

    # ── Optimiser ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── LR scheduler (optional) ─────────────────────────────
    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )
        logger.info("Cosine-annealing LR scheduler enabled.")

    # ── Resume ───────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume and os.path.isfile(args.resume):
        ckpt = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))

    # ══════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 65)
    logger.info(
        "Starting temporal training | backbone=%s  aggregator=%s  "
        "seq_len=%d  epochs=%d  batch=%d  lr=%.1e",
        args.backbone, args.aggregator, args.seq_len,
        args.epochs, args.batch_size, args.lr,
    )
    logger.info("=" * 65)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────
        t_loss, t_met = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # ── Validate ─────────────────────────────────────────
        v_loss, v_met, v_true, v_pred = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        logger.info(
            "Epoch %2d/%d  [%.1fs]  "
            "Train — loss=%.4f  acc=%.3f  f1=%.3f  |  "
            "Val — loss=%.4f  acc=%.3f  f1=%.3f  |  lr=%.2e",
            epoch + 1, args.epochs, elapsed,
            t_loss, t_met["accuracy"], t_met["f1"],
            v_loss, v_met["accuracy"], v_met["f1"],
            current_lr,
        )

        # ── Save best ────────────────────────────────────────
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_checkpoint(args.checkpoint, model, optimizer, epoch, v_loss, v_met)

    # ══════════════════════════════════════════════════════════
    # Final evaluation
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 65)
    logger.info("Training complete. Loading best checkpoint for final eval...")

    if os.path.isfile(args.checkpoint):
        load_checkpoint(args.checkpoint, model, device=device)

    model.to(device)
    final_loss, final_met, y_true, y_pred = validate(model, val_loader, criterion, device)

    logger.info("-" * 65)
    logger.info("Final Validation Results:")
    logger.info("  Loss:      %.4f", final_loss)
    logger.info("  Accuracy:  %.4f", final_met["accuracy"])
    logger.info("  F1 Score:  %.4f", final_met["f1"])
    logger.info("  Precision: %.4f", final_met["precision"])
    logger.info("  Recall:    %.4f", final_met["recall"])
    logger.info("-" * 65)

    print_confusion_matrix(y_true, y_pred)

    logger.info("=" * 65)
    logger.info("Done. Best checkpoint: %s", args.checkpoint)


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
