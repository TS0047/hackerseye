"""
train_antispoof.py — Phase 3: Baseline CNN training for replay anti-spoofing.

Trains a pretrained ResNet-18 with a binary classifier head to distinguish
real faces from spoof (screen / replay) faces using crops produced by
Phase 1 (extract_faces.py) and loaded via Phase 2 (antispoof_dataset.py).

Features:
  - Pretrained ResNet-18 backbone (ImageNet weights, fine-tuned)
  - Binary classification: real (0) vs spoof (1)
  - BCEWithLogitsLoss (handles class imbalance via pos_weight)
  - AdamW optimiser with optional cosine-annealing LR scheduler
  - Saves best checkpoint (lowest val loss) to disk
  - Per-epoch logging: loss, accuracy, F1, precision, recall
  - Final evaluation on validation set with confusion matrix

Usage:
    # Minimal — uses defaults from .env and CLI
    python train_antispoof.py --csv data/frames/my_dataset/labels.csv \\
                              --root data/frames/my_dataset

    # Full control
    python train_antispoof.py --csv data/frames/my_dataset/labels.csv \\
                              --root data/frames/my_dataset \\
                              --epochs 25 --batch-size 32 --lr 1e-4 \\
                              --val-split 0.2 --checkpoint checkpoints/best.pt \\
                              --scheduler --freeze-backbone
"""

# ═══════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════
import argparse
import os
import sys
import time
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ── Metrics ──────────────────────────────────────────────────
# sklearn is optional; we provide lightweight fallback metrics
try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Phase 2 dataset + transforms ────────────────────────────
from antispoof_dataset import (
    AntiSpoofDataset,
    get_transforms,
    load_csv,
    LABEL_MAP,
)

# ═══════════════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. Model builder — ResNet-18 → binary classifier
# ═══════════════════════════════════════════════════════════════
def build_model(
    freeze_backbone: bool = False,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a ResNet-18 with ImageNet weights and replace the final FC layer
    with a single-output binary classifier head.

    Parameters
    ----------
    freeze_backbone : bool
        If True, freeze all layers except the final classifier head.
        This speeds up training when fine-tuning on a small dataset.
    pretrained : bool
        Load ImageNet weights (default True).

    Returns
    -------
    nn.Module
        The modified ResNet-18 model.
    """
    # torchvision >= 0.13 uses 'weights' kwarg instead of deprecated 'pretrained'
    import torchvision.models as models

    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        logger.info("Loading ResNet-18 with ImageNet weights.")
    else:
        weights = None
        logger.info("Loading ResNet-18 without pre-trained weights.")

    model = models.resnet18(weights=weights)

    # ── Optionally freeze backbone layers ────────────────────
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Backbone layers frozen — only head will be trained.")

    # ── Replace final fully-connected layer ──────────────────
    # Original: Linear(512, 1000) → we want Linear(512, 1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)  # single logit for BCEWithLogitsLoss

    logger.info(
        "Classifier head: Linear(%d, 1)  |  Total params: %s  |  Trainable: %s",
        in_features,
        f"{sum(p.numel() for p in model.parameters()):,}",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )
    return model


# ═══════════════════════════════════════════════════════════════
# 2. Fallback metrics (when sklearn is unavailable)
# ═══════════════════════════════════════════════════════════════
def _accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Simple accuracy."""
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / max(len(y_true), 1)


def _f1(y_true: List[int], y_pred: List[int]) -> float:
    """Binary F1 score (positive class = 1 = spoof)."""
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _precision(y_true: List[int], y_pred: List[int]) -> float:
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    return tp / max(tp + fp, 1)


def _recall(y_true: List[int], y_pred: List[int]) -> float:
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    return tp / max(tp + fn, 1)


def _confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """2×2 confusion matrix: [[TN, FP], [FN, TP]]."""
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    return np.array([[tn, fp], [fn, tp]])


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Compute accuracy, F1, precision, recall.

    Uses sklearn when available; falls back to lightweight implementations.
    """
    if HAS_SKLEARN:
        return {
            "accuracy":  accuracy_score(y_true, y_pred),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
        }
    else:
        return {
            "accuracy":  _accuracy(y_true, y_pred),
            "f1":        _f1(y_true, y_pred),
            "precision": _precision(y_true, y_pred),
            "recall":    _recall(y_true, y_pred),
        }


def print_confusion_matrix(y_true: List[int], y_pred: List[int]) -> None:
    """Pretty-print a 2×2 confusion matrix."""
    if HAS_SKLEARN:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    else:
        cm = _confusion_matrix(y_true, y_pred)

    logger.info("Confusion Matrix:")
    logger.info("                 Predicted")
    logger.info("              Real    Spoof")
    logger.info("  Actual Real  %4d    %4d", cm[0, 0], cm[0, 1])
    logger.info("  Actual Spoof %4d    %4d", cm[1, 0], cm[1, 1])


# ═══════════════════════════════════════════════════════════════
# 3. Training loop — one epoch
# ═══════════════════════════════════════════════════════════════
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        The classifier model.
    loader : DataLoader
        Training dataloader.
    criterion : nn.Module
        Loss function (BCEWithLogitsLoss).
    optimizer : torch.optim.Optimizer
        Optimiser instance.
    device : torch.device
        Target device (cpu or cuda).

    Returns
    -------
    Tuple[float, Dict[str, float]]
        (average loss, metrics dict with accuracy/f1/precision/recall)
    """
    model.train()
    running_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []

    for batch_idx, (images, labels) in enumerate(loader):
        # images: (B, C, H, W),  labels: (B,) int
        images = images.to(device)
        labels = labels.float().to(device)  # BCEWithLogitsLoss expects float targets

        # ── Forward pass ─────────────────────────────────────
        logits = model(images).squeeze(1)   # (B,) raw logits
        loss = criterion(logits, labels)

        # ── Backward pass ────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Track statistics ─────────────────────────────────
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        all_labels.extend(labels.long().cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / max(len(all_labels), 1)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


# ═══════════════════════════════════════════════════════════════
# 4. Validation loop — one epoch
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], List[int], List[int]]:
    """
    Evaluate model on validation set.

    Returns
    -------
    Tuple[float, Dict, List[int], List[int]]
        (avg_loss, metrics, all_true_labels, all_pred_labels)
    """
    model.eval()
    running_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels_f = labels.float().to(device)

        logits = model(images).squeeze(1)
        loss = criterion(logits, labels_f)

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        all_labels.extend(labels.long().cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / max(len(all_labels), 1)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_labels, all_preds


# ═══════════════════════════════════════════════════════════════
# 5. Checkpoint helpers
# ═══════════════════════════════════════════════════════════════
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    metrics: Dict[str, float],
) -> None:
    """Save model + optimiser state to a .pt file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "metrics": metrics,
        },
        path,
    )
    logger.info("💾  Checkpoint saved → %s  (val_loss=%.4f)", path, val_loss)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Restore model (and optionally optimiser) from a checkpoint.

    Returns the full checkpoint dict so callers can read epoch / metrics.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(
        "✅  Loaded checkpoint from %s  (epoch %d, val_loss=%.4f)",
        path, ckpt.get("epoch", -1), ckpt.get("val_loss", float("inf")),
    )
    return ckpt


# ═══════════════════════════════════════════════════════════════
# 6. CLI argument parser
# ═══════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Phase 3 — Train baseline anti-spoof CNN (ResNet-18).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ─────────────────────────────────────────────────
    p.add_argument(
        "--csv", required=True,
        help="Path to labels.csv from Phase 1 extraction.",
    )
    p.add_argument(
        "--root", required=True,
        help="Root directory containing the face crops (same as Phase 1 OUTPUT_DIR/dataset).",
    )

    # ── Training hyper-parameters ────────────────────────────
    p.add_argument("--epochs",       type=int,   default=20,    help="Number of training epochs.")
    p.add_argument("--batch-size",   type=int,   default=32,    help="Mini-batch size.")
    p.add_argument("--lr",           type=float, default=1e-4,  help="Initial learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4,  help="AdamW weight decay.")
    p.add_argument("--val-split",    type=float, default=0.2,   help="Fraction of data for validation.")

    # ── Model options ────────────────────────────────────────
    p.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze ResNet backbone and only train the classifier head.",
    )
    p.add_argument(
        "--no-pretrained", action="store_true",
        help="Initialise ResNet-18 from scratch (no ImageNet weights).",
    )

    # ── Scheduler / checkpoint ───────────────────────────────
    p.add_argument(
        "--scheduler", action="store_true",
        help="Enable cosine-annealing LR scheduler.",
    )
    p.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_antispoof.pt",
        help="Where to save the best model checkpoint.",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to resume training from.",
    )

    # ── Misc ────────────────────────────────────────────────
    p.add_argument("--seed",    type=int, default=42,  help="Random seed.")
    p.add_argument("--workers", type=int, default=0,   help="DataLoader worker processes.")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# 7. Main entry point
# ═══════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_args()

    # ── Reproducibility ──────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device selection ─────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.warning(
            "CUDA not available — training on CPU (this will be slow). "
            "Install CUDA-enabled PyTorch: "
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        )

    # ── Build datasets ───────────────────────────────────────
    logger.info("Loading CSV: %s", args.csv)
    full_dataset = AntiSpoofDataset(
        csv_path=args.csv,
        root_dir=args.root,
        transform=None,  # we set transforms after splitting
    )
    total = len(full_dataset)
    if total == 0:
        logger.error("Dataset is empty.  Run extract_faces.py first.")
        sys.exit(1)

    # ── Train / val split ────────────────────────────────────
    val_size   = max(1, int(total * args.val_split))
    train_size = total - val_size

    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Apply different transforms to train vs val via wrapper
    train_ds.dataset = AntiSpoofDataset(
        csv_path=args.csv,
        root_dir=args.root,
        transform=get_transforms(train=True),
    )
    # Re-split so val_ds also points to the eval-transform dataset
    # We use Subset indices directly to keep splits consistent
    val_dataset_eval = AntiSpoofDataset(
        csv_path=args.csv,
        root_dir=args.root,
        transform=get_transforms(train=False),
    )

    # Wrap subsets with correct transforms using a thin wrapper
    class _SubsetWithTransform(torch.utils.data.Dataset):
        """Wraps random_split indices with a specific AntiSpoofDataset."""
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    train_loader_ds = _SubsetWithTransform(
        AntiSpoofDataset(args.csv, args.root, transform=get_transforms(train=True)),
        train_ds.indices,
    )
    val_loader_ds = _SubsetWithTransform(
        val_dataset_eval,
        val_ds.indices,
    )

    logger.info("Split: %d train / %d val  (%.0f%% val)", train_size, val_size, args.val_split * 100)

    # ── Label distribution ───────────────────────────────────
    counts = full_dataset.get_label_counts()
    logger.info("Label counts — real: %d, spoof: %d", counts.get("real", 0), counts.get("spoof", 0))

    # ── DataLoaders ──────────────────────────────────────────
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
    model = build_model(
        freeze_backbone=args.freeze_backbone,
        pretrained=not args.no_pretrained,
    ).to(device)

    # ── Loss function with class-weight ──────────────────────
    # pos_weight compensates for class imbalance:
    # pos_weight = num_real / num_spoof  (if spoof is positive class)
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
        logger.info("Cosine-annealing LR scheduler enabled (T_max=%d).", args.epochs)

    # ── Resume from checkpoint ───────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume and os.path.isfile(args.resume):
        ckpt = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        logger.info("Resuming from epoch %d.", start_epoch)

    # ══════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Starting training  |  epochs=%d  batch=%d  lr=%.1e",
                args.epochs, args.batch_size, args.lr)
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # ── Train ────────────────────────────────────────────
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )

        # ── Validate ─────────────────────────────────────────
        val_loss, val_metrics, val_true, val_pred = validate(
            model, val_loader, criterion, device,
        )

        # ── Scheduler step ───────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - epoch_start

        # ── Log epoch summary ────────────────────────────────
        logger.info(
            "Epoch %2d/%d  [%.1fs]  "
            "Train — loss=%.4f  acc=%.3f  f1=%.3f  |  "
            "Val — loss=%.4f  acc=%.3f  f1=%.3f  |  lr=%.2e",
            epoch + 1, args.epochs, elapsed,
            train_loss, train_metrics["accuracy"], train_metrics["f1"],
            val_loss,   val_metrics["accuracy"],   val_metrics["f1"],
            current_lr,
        )

        # ── Save best checkpoint ─────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                args.checkpoint, model, optimizer, epoch, val_loss, val_metrics,
            )

    # ══════════════════════════════════════════════════════════
    # Final evaluation on validation set
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Training complete.  Loading best checkpoint for final eval...")

    if os.path.isfile(args.checkpoint):
        load_checkpoint(args.checkpoint, model, device=device)

    model.to(device)
    final_loss, final_metrics, y_true, y_pred = validate(
        model, val_loader, criterion, device,
    )

    logger.info("-" * 60)
    logger.info("Final Validation Results:")
    logger.info("  Loss:      %.4f", final_loss)
    logger.info("  Accuracy:  %.4f", final_metrics["accuracy"])
    logger.info("  F1 Score:  %.4f", final_metrics["f1"])
    logger.info("  Precision: %.4f", final_metrics["precision"])
    logger.info("  Recall:    %.4f", final_metrics["recall"])
    logger.info("-" * 60)

    print_confusion_matrix(y_true, y_pred)

    logger.info("=" * 60)
    logger.info("Done. Best checkpoint: %s", args.checkpoint)


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
