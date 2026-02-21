"""
test_train_temporal.py — Phase 4 unit tests for the temporal anti-spoofing model.

Tests cover:
  - Backbone construction (ResNet-18, MobileNetV3) + feature dims
  - Temporal aggregators (avg pool, max pool, LSTM) shapes
  - Full TemporalAntiSpoofModel forward pass shapes
  - Backbone freezing
  - All aggregator variants
  - Training loop (one epoch on synthetic sequence data)
  - Validation loop
  - Checkpoint save / load with temporal model
  - CLI argument parsing
  - Mini end-to-end training run

Run:
    pytest test_train_temporal.py -v
"""

import os
import tempfile
import shutil
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_temporal import (
    _build_backbone,
    TemporalAvgPool,
    TemporalMaxPool,
    TemporalLSTM,
    TemporalAntiSpoofModel,
    train_one_epoch,
    validate,
    parse_args,
)

from train_antispoof import save_checkpoint, load_checkpoint


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def seq_batch():
    """Synthetic batch: 4 clips, 8 frames each, 224×224 RGB."""
    B, T, C, H, W = 4, 8, 3, 224, 224
    sequences = torch.randn(B, T, C, H, W)
    labels = torch.tensor([0, 1, 1, 0])
    return sequences, labels


@pytest.fixture
def small_seq_batch():
    """Smaller batch for speed: 2 clips, 4 frames, 64×64."""
    B, T, C, H, W = 2, 4, 3, 64, 64
    sequences = torch.randn(B, T, C, H, W)
    labels = torch.tensor([0, 1])
    return sequences, labels


@pytest.fixture
def seq_loader(seq_batch):
    """DataLoader wrapping the synthetic sequence batch."""
    sequences, labels = seq_batch
    ds = TensorDataset(sequences, labels)
    return DataLoader(ds, batch_size=2, shuffle=False)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 1. Backbone tests
# ═══════════════════════════════════════════════════════════════

class TestBackbone:

    def test_resnet18_feature_dim(self):
        backbone, dim = _build_backbone("resnet18", pretrained=False, freeze=False)
        assert dim == 512

    def test_resnet18_output_shape(self):
        backbone, dim = _build_backbone("resnet18", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = backbone(x)
        assert out.shape == (2, 512)

    def test_mobilenet_feature_dim(self):
        backbone, dim = _build_backbone("mobilenet", pretrained=False)
        assert dim == 576

    def test_mobilenet_output_shape(self):
        backbone, dim = _build_backbone("mobilenet", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = backbone(x)
        assert out.shape == (2, 576)

    def test_freeze_backbone(self):
        backbone, _ = _build_backbone("resnet18", pretrained=False, freeze=True)
        trainable = [p for p in backbone.parameters() if p.requires_grad]
        assert len(trainable) == 0

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            _build_backbone("vgg16", pretrained=False)


# ═══════════════════════════════════════════════════════════════
# 2. Temporal aggregator tests
# ═══════════════════════════════════════════════════════════════

class TestTemporalAggregators:

    def test_avg_pool_shape(self):
        agg = TemporalAvgPool()
        x = torch.randn(4, 8, 512)   # (B, T, D)
        out = agg(x)
        assert out.shape == (4, 512)  # (B, D)

    def test_avg_pool_values(self):
        """Average pool should equal manual mean."""
        agg = TemporalAvgPool()
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        out = agg(x)
        expected = torch.tensor([[2.0, 3.0]])
        torch.testing.assert_close(out, expected)

    def test_max_pool_shape(self):
        agg = TemporalMaxPool()
        x = torch.randn(4, 8, 512)
        out = agg(x)
        assert out.shape == (4, 512)

    def test_max_pool_values(self):
        agg = TemporalMaxPool()
        x = torch.tensor([[[1.0, 4.0], [3.0, 2.0]]])
        out = agg(x)
        expected = torch.tensor([[3.0, 4.0]])
        torch.testing.assert_close(out, expected)

    def test_lstm_shape(self):
        agg = TemporalLSTM(input_dim=512, hidden_size=128, num_layers=1)
        x = torch.randn(4, 8, 512)
        out = agg(x)
        assert out.shape == (4, 256)  # hidden * 2 (bidirectional)

    def test_lstm_output_dim_property(self):
        agg = TemporalLSTM(input_dim=256, hidden_size=64)
        assert agg.output_dim == 128  # 64 * 2

    def test_lstm_multi_layer(self):
        agg = TemporalLSTM(input_dim=512, hidden_size=64, num_layers=2, dropout=0.1)
        x = torch.randn(2, 4, 512)
        out = agg(x)
        assert out.shape == (2, 128)


# ═══════════════════════════════════════════════════════════════
# 3. Full model tests
# ═══════════════════════════════════════════════════════════════

class TestTemporalModel:

    def test_lstm_forward_shape(self, device):
        model = TemporalAntiSpoofModel(
            backbone_name="resnet18", aggregator="lstm",
            pretrained=False, lstm_hidden=64,
        ).to(device)
        x = torch.randn(2, 4, 3, 224, 224, device=device)
        out = model(x)
        assert out.shape == (2, 1)

    def test_avg_forward_shape(self, device):
        model = TemporalAntiSpoofModel(
            backbone_name="resnet18", aggregator="avg", pretrained=False,
        ).to(device)
        x = torch.randn(2, 4, 3, 224, 224, device=device)
        out = model(x)
        assert out.shape == (2, 1)

    def test_max_forward_shape(self, device):
        model = TemporalAntiSpoofModel(
            backbone_name="resnet18", aggregator="max", pretrained=False,
        ).to(device)
        x = torch.randn(2, 4, 3, 224, 224, device=device)
        out = model(x)
        assert out.shape == (2, 1)

    def test_mobilenet_lstm(self, device):
        model = TemporalAntiSpoofModel(
            backbone_name="mobilenet", aggregator="lstm",
            pretrained=False, lstm_hidden=64,
        ).to(device)
        x = torch.randn(2, 4, 3, 224, 224, device=device)
        out = model(x)
        assert out.shape == (2, 1)

    def test_frozen_backbone_trainable_params(self, device):
        model = TemporalAntiSpoofModel(
            backbone_name="resnet18", aggregator="lstm",
            pretrained=False, freeze_backbone=True, lstm_hidden=64,
        ).to(device)
        # Backbone frozen → only LSTM + classifier head trainable
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert all("backbone" not in n for n in trainable), \
            f"Backbone params should be frozen, but found: {[n for n in trainable if 'backbone' in n]}"
        assert len(trainable) > 0, "At least temporal + classifier should be trainable"

    def test_unknown_aggregator_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregator"):
            TemporalAntiSpoofModel(aggregator="attention", pretrained=False)


# ═══════════════════════════════════════════════════════════════
# 4. Training loop tests
# ═══════════════════════════════════════════════════════════════

class TestTrainOneEpoch:

    def test_returns_loss_and_metrics(self, seq_loader, device):
        model = TemporalAntiSpoofModel(
            backbone_name="resnet18", aggregator="avg", pretrained=False,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        loss, metrics = train_one_epoch(model, seq_loader, criterion, optimizer, device)
        assert isinstance(loss, float) and loss >= 0
        assert "accuracy" in metrics and "f1" in metrics

    def test_metrics_in_range(self, seq_loader, device):
        model = TemporalAntiSpoofModel(
            backbone_name="resnet18", aggregator="avg", pretrained=False,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        _, metrics = train_one_epoch(model, seq_loader, criterion, optimizer, device)
        for key in ("accuracy", "f1", "precision", "recall"):
            assert 0.0 <= metrics[key] <= 1.0


# ═══════════════════════════════════════════════════════════════
# 5. Validation loop tests
# ═══════════════════════════════════════════════════════════════

class TestValidate:

    def test_returns_four_values(self, seq_loader, device):
        model = TemporalAntiSpoofModel(
            backbone_name="resnet18", aggregator="avg", pretrained=False,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()

        result = validate(model, seq_loader, criterion, device)
        assert len(result) == 4

    def test_label_counts_match(self, seq_loader, device):
        model = TemporalAntiSpoofModel(
            backbone_name="resnet18", aggregator="avg", pretrained=False,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()

        _, _, y_true, y_pred = validate(model, seq_loader, criterion, device)
        assert len(y_true) == 4   # 4 samples in seq_batch
        assert len(y_pred) == 4


# ═══════════════════════════════════════════════════════════════
# 6. Checkpoint tests
# ═══════════════════════════════════════════════════════════════

class TestCheckpointTemporal:

    def test_save_load_roundtrip(self, tmp_dir, device):
        model = TemporalAntiSpoofModel(
            backbone_name="resnet18", aggregator="lstm",
            pretrained=False, lstm_hidden=32,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        path = os.path.join(tmp_dir, "temporal.pt")
        save_checkpoint(path, model, optimizer, epoch=3, val_loss=0.4, metrics={"f1": 0.7})
        assert os.path.isfile(path)

        model2 = TemporalAntiSpoofModel(
            backbone_name="resnet18", aggregator="lstm",
            pretrained=False, lstm_hidden=32,
        ).to(device)
        ckpt = load_checkpoint(path, model2, device=device)
        assert ckpt["epoch"] == 3
        assert abs(ckpt["val_loss"] - 0.4) < 1e-6


# ═══════════════════════════════════════════════════════════════
# 7. CLI tests
# ═══════════════════════════════════════════════════════════════

class TestParseArgs:

    def test_required_args(self):
        with mock.patch("sys.argv", ["prog", "--csv", "a.csv", "--root", "d/"]):
            args = parse_args()
        assert args.csv == "a.csv"
        assert args.root == "d/"

    def test_defaults(self):
        with mock.patch("sys.argv", ["prog", "--csv", "x.csv", "--root", "r/"]):
            args = parse_args()
        assert args.seq_len == 8
        assert args.stride == 4
        assert args.backbone == "resnet18"
        assert args.aggregator == "lstm"
        assert args.batch_size == 8
        assert args.epochs == 20

    def test_custom_overrides(self):
        with mock.patch("sys.argv", [
            "prog", "--csv", "x.csv", "--root", "r/",
            "--backbone", "mobilenet", "--aggregator", "avg",
            "--seq-len", "16", "--stride", "8",
            "--epochs", "50", "--batch-size", "4",
            "--freeze-backbone", "--scheduler",
        ]):
            args = parse_args()
        assert args.backbone == "mobilenet"
        assert args.aggregator == "avg"
        assert args.seq_len == 16
        assert args.freeze_backbone is True
        assert args.scheduler is True


# ═══════════════════════════════════════════════════════════════
# 8. Integration: mini training run
# ═══════════════════════════════════════════════════════════════

class TestMiniTemporalTraining:

    def test_full_pipeline(self, tmp_dir, device):
        """Build → train 2 epochs → validate → save/load."""
        B, T = 6, 4
        sequences = torch.randn(B, T, 3, 224, 224)
        labels = torch.tensor([0, 1, 1, 0, 1, 0])
        ds = TensorDataset(sequences, labels)
        loader = DataLoader(ds, batch_size=2, shuffle=True)

        model = TemporalAntiSpoofModel(
            backbone_name="resnet18", aggregator="avg", pretrained=False,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        ckpt_path = os.path.join(tmp_dir, "mini_temporal.pt")
        best_val = float("inf")

        for epoch in range(2):
            t_loss, _ = train_one_epoch(model, loader, criterion, optimizer, device)
            v_loss, v_met, _, _ = validate(model, loader, criterion, device)
            assert isinstance(t_loss, float)
            assert isinstance(v_loss, float)

            if v_loss < best_val:
                best_val = v_loss
                save_checkpoint(ckpt_path, model, optimizer, epoch, v_loss, v_met)

        assert os.path.isfile(ckpt_path)

        # Reload
        model2 = TemporalAntiSpoofModel(
            backbone_name="resnet18", aggregator="avg", pretrained=False,
        ).to(device)
        load_checkpoint(ckpt_path, model2, device=device)
        v2, _, _, _ = validate(model2, loader, criterion, device)
        assert isinstance(v2, float)
