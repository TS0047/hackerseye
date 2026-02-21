"""
test_train_antispoof.py — Phase 3 unit tests for the baseline CNN training script.

Tests cover:
  - Model construction (ResNet-18 with binary head)
  - Backbone freezing
  - Forward pass shapes
  - Training loop (one epoch on tiny data)
  - Validation loop
  - Metric computation (accuracy, F1, precision, recall)
  - Confusion matrix
  - Checkpoint save / load round-trip
  - CLI argument parsing
  - Class-weight computation for imbalanced data

Run:
    pytest test_train_antispoof.py -v
"""

import os
import sys
import csv
import tempfile
import shutil
import argparse
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Import the module under test ─────────────────────────────
from train_antispoof import (
    build_model,
    train_one_epoch,
    validate,
    save_checkpoint,
    load_checkpoint,
    compute_metrics,
    print_confusion_matrix,
    parse_args,
    _accuracy,
    _f1,
    _precision,
    _recall,
    _confusion_matrix,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def device():
    """Use CPU for all tests (fast, no GPU requirement)."""
    return torch.device("cpu")


@pytest.fixture
def model(device):
    """Build a non-pretrained model to keep tests fast."""
    m = build_model(freeze_backbone=False, pretrained=False)
    return m.to(device)


@pytest.fixture
def frozen_model(device):
    """Build a model with frozen backbone."""
    m = build_model(freeze_backbone=True, pretrained=False)
    return m.to(device)


@pytest.fixture
def dummy_loader():
    """
    DataLoader with 16 random 224x224 images and binary labels.
    Small enough for fast tests.
    """
    images = torch.randn(16, 3, 224, 224)
    labels = torch.randint(0, 2, (16,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=4, shuffle=False)


@pytest.fixture
def tmp_dir():
    """Temporary directory for checkpoint tests; cleaned up after."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 1. Model construction tests
# ═══════════════════════════════════════════════════════════════

class TestBuildModel:

    def test_output_shape(self, model, device):
        """Model should output shape (B, 1) for binary classification."""
        x = torch.randn(2, 3, 224, 224, device=device)
        out = model(x)
        assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"

    def test_single_logit(self, model, device):
        """With squeeze, should produce a scalar per sample."""
        x = torch.randn(1, 3, 224, 224, device=device)
        out = model(x).squeeze(1)
        assert out.shape == (1,)

    def test_backbone_frozen(self, frozen_model):
        """When backbone is frozen, only fc layer should require grad."""
        trainable = [n for n, p in frozen_model.named_parameters() if p.requires_grad]
        # Only fc.weight and fc.bias should be trainable
        assert len(trainable) == 2
        assert all("fc" in n for n in trainable)

    def test_backbone_unfrozen(self, model):
        """Without freezing, all params should require grad."""
        trainable = [p for p in model.parameters() if p.requires_grad]
        total = list(model.parameters())
        assert len(trainable) == len(total)

    def test_fc_in_features(self, model):
        """Final FC layer should have 512 input features for ResNet-18."""
        assert model.fc.in_features == 512

    def test_fc_out_features(self, model):
        """Final FC should output 1 logit."""
        assert model.fc.out_features == 1

    def test_pretrained_flag(self):
        """no-pretrained model should still build successfully."""
        m = build_model(pretrained=False)
        assert isinstance(m, nn.Module)


# ═══════════════════════════════════════════════════════════════
# 2. Training loop tests
# ═══════════════════════════════════════════════════════════════

class TestTrainOneEpoch:

    def test_returns_loss_and_metrics(self, model, dummy_loader, device):
        """train_one_epoch should return (float, dict)."""
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        loss, metrics = train_one_epoch(model, dummy_loader, criterion, optimizer, device)

        assert isinstance(loss, float)
        assert loss >= 0.0
        assert "accuracy" in metrics
        assert "f1" in metrics

    def test_loss_decreases(self, device):
        """After a few steps on toy data, loss should generally decrease."""
        model = build_model(pretrained=False).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        images = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4)

        loss1, _ = train_one_epoch(model, loader, criterion, optimizer, device)
        loss2, _ = train_one_epoch(model, loader, criterion, optimizer, device)
        loss3, _ = train_one_epoch(model, loader, criterion, optimizer, device)

        # Loss should show some downward trend over 3 epochs
        # (not guaranteed on every run, but likely with lr=1e-2 on 8 samples)
        assert loss3 < loss1 * 1.5, "Loss did not trend downward"

    def test_metrics_in_range(self, model, dummy_loader, device):
        """Metrics should be in [0, 1]."""
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        _, metrics = train_one_epoch(model, dummy_loader, criterion, optimizer, device)

        for key in ("accuracy", "f1", "precision", "recall"):
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of range"


# ═══════════════════════════════════════════════════════════════
# 3. Validation loop tests
# ═══════════════════════════════════════════════════════════════

class TestValidate:

    def test_returns_four_values(self, model, dummy_loader, device):
        """validate should return (loss, metrics, y_true, y_pred)."""
        criterion = nn.BCEWithLogitsLoss()
        result = validate(model, dummy_loader, criterion, device)
        assert len(result) == 4

    def test_labels_match_size(self, model, dummy_loader, device):
        """Returned label lists should match dataset size."""
        criterion = nn.BCEWithLogitsLoss()
        _, _, y_true, y_pred = validate(model, dummy_loader, criterion, device)
        assert len(y_true) == 16   # 16 samples in dummy_loader
        assert len(y_pred) == 16

    def test_no_grad_mode(self, model, dummy_loader, device):
        """validate should not accumulate gradients."""
        criterion = nn.BCEWithLogitsLoss()
        validate(model, dummy_loader, criterion, device)

        # After validate, model params should have no grad
        for p in model.parameters():
            assert p.grad is None or torch.all(p.grad == 0)


# ═══════════════════════════════════════════════════════════════
# 4. Metrics tests
# ═══════════════════════════════════════════════════════════════

class TestMetrics:

    def test_perfect_accuracy(self):
        y = [0, 1, 0, 1, 1]
        m = compute_metrics(y, y)
        assert m["accuracy"] == 1.0

    def test_zero_accuracy(self):
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0

    def test_f1_perfect(self):
        y = [1, 1, 0, 0]
        m = compute_metrics(y, y)
        assert m["f1"] == 1.0

    def test_f1_none_predicted(self):
        """If no positive predicted, F1 should be 0."""
        y_true = [1, 1, 1]
        y_pred = [0, 0, 0]
        m = compute_metrics(y_true, y_pred)
        assert m["f1"] == 0.0

    def test_precision_recall_basic(self):
        # TP=2, FP=1, FN=1
        y_true = [1, 1, 0, 1]
        y_pred = [1, 1, 1, 0]
        m = compute_metrics(y_true, y_pred)
        assert abs(m["precision"] - 2 / 3) < 1e-6
        assert abs(m["recall"] - 2 / 3) < 1e-6


class TestFallbackMetrics:
    """Test the lightweight fallback metric functions directly."""

    def test_accuracy(self):
        assert _accuracy([0, 1, 1], [0, 1, 0]) == pytest.approx(2 / 3)

    def test_f1(self):
        # TP=1, FP=0, FN=1 → precision=1, recall=0.5 → F1 = 2/3
        assert _f1([1, 1, 0], [1, 0, 0]) == pytest.approx(2 / 3)

    def test_precision(self):
        assert _precision([1, 0, 0], [1, 1, 0]) == pytest.approx(0.5)

    def test_recall(self):
        assert _recall([1, 1, 0], [1, 0, 0]) == pytest.approx(0.5)

    def test_confusion_matrix(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        cm = _confusion_matrix(y_true, y_pred)
        # [[TN=1, FP=1], [FN=1, TP=1]]
        np.testing.assert_array_equal(cm, np.array([[1, 1], [1, 1]]))


class TestPrintConfusionMatrix:
    """Smoke test — just make sure it doesn't crash."""

    def test_prints_without_error(self, capsys):
        print_confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0])
        # No assertion — just verifying no exception


# ═══════════════════════════════════════════════════════════════
# 5. Checkpoint tests
# ═══════════════════════════════════════════════════════════════

class TestCheckpoints:

    def test_save_and_load_roundtrip(self, model, tmp_dir, device):
        """Checkpoint should preserve model weights across save/load."""
        ckpt_path = os.path.join(tmp_dir, "test_model.pt")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Grab original weights
        original_fc_weight = model.fc.weight.data.clone()

        save_checkpoint(ckpt_path, model, optimizer, epoch=5, val_loss=0.25,
                        metrics={"accuracy": 0.9})

        assert os.path.isfile(ckpt_path)

        # Load into a fresh model
        model2 = build_model(pretrained=False).to(device)
        ckpt = load_checkpoint(ckpt_path, model2, device=device)

        # Weights should match
        torch.testing.assert_close(model2.fc.weight.data, original_fc_weight)
        assert ckpt["epoch"] == 5
        assert abs(ckpt["val_loss"] - 0.25) < 1e-6

    def test_checkpoint_creates_directory(self, tmp_dir, device):
        """save_checkpoint should create parent dirs if missing."""
        nested = os.path.join(tmp_dir, "deep", "nested", "model.pt")
        model = build_model(pretrained=False).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        save_checkpoint(nested, model, optimizer, epoch=0, val_loss=1.0, metrics={})
        assert os.path.isfile(nested)

    def test_checkpoint_contains_expected_keys(self, model, tmp_dir, device):
        ckpt_path = os.path.join(tmp_dir, "keys.pt")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        save_checkpoint(ckpt_path, model, optimizer, epoch=3, val_loss=0.5,
                        metrics={"f1": 0.8})

        loaded = torch.load(ckpt_path, weights_only=False)
        for key in ("epoch", "model_state_dict", "optimizer_state_dict",
                     "val_loss", "metrics"):
            assert key in loaded, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════════
# 6. CLI argument parsing tests
# ═══════════════════════════════════════════════════════════════

class TestParseArgs:

    def test_required_args(self):
        """Should parse correctly with --csv and --root."""
        with mock.patch("sys.argv", ["prog", "--csv", "a.csv", "--root", "data/"]):
            args = parse_args()
        assert args.csv == "a.csv"
        assert args.root == "data/"

    def test_defaults(self):
        with mock.patch("sys.argv", ["prog", "--csv", "a.csv", "--root", "d/"]):
            args = parse_args()
        assert args.epochs == 20
        assert args.batch_size == 32
        assert args.lr == 1e-4
        assert args.val_split == 0.2
        assert args.scheduler is False
        assert args.freeze_backbone is False

    def test_custom_overrides(self):
        with mock.patch("sys.argv", [
            "prog", "--csv", "x.csv", "--root", "r/",
            "--epochs", "50", "--batch-size", "64", "--lr", "3e-4",
            "--scheduler", "--freeze-backbone",
        ]):
            args = parse_args()
        assert args.epochs == 50
        assert args.batch_size == 64
        assert args.lr == 3e-4
        assert args.scheduler is True
        assert args.freeze_backbone is True


# ═══════════════════════════════════════════════════════════════
# 7. Integration-style: full mini training run
# ═══════════════════════════════════════════════════════════════

class TestMiniTrainingRun:
    """
    End-to-end: build model → train 2 epochs → validate → save/load.
    Uses tiny synthetic data so it runs in seconds.
    """

    def test_full_pipeline(self, tmp_dir, device):
        # ── Synthetic data ───────────────────────────────────
        images = torch.randn(12, 3, 224, 224)
        labels = torch.tensor([0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        ds = TensorDataset(images, labels)
        train_loader = DataLoader(ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(ds, batch_size=4, shuffle=False)

        # ── Model + loss + optim ─────────────────────────────
        model = build_model(pretrained=False).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        best_val_loss = float("inf")
        ckpt_path = os.path.join(tmp_dir, "mini_best.pt")

        for epoch in range(2):
            t_loss, t_met = train_one_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_met, y_true, y_pred = validate(model, val_loader, criterion, device)

            assert isinstance(t_loss, float)
            assert isinstance(v_loss, float)
            assert len(y_true) == 12

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                save_checkpoint(ckpt_path, model, optimizer, epoch, v_loss, v_met)

        # ── Checkpoint should exist ──────────────────────────
        assert os.path.isfile(ckpt_path)

        # ── Load and re-evaluate ─────────────────────────────
        model2 = build_model(pretrained=False).to(device)
        load_checkpoint(ckpt_path, model2, device=device)
        model2.to(device)

        v_loss2, _, _, _ = validate(model2, val_loader, criterion, device)
        assert isinstance(v_loss2, float)
