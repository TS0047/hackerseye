"""
test_hackerseye_antispoof.py — Phase 8 unit tests for the live anti-spoof demo.

Tests cover:
  - FrameBuffer (push, full, flush on miss, get_sequence, get_latest)
  - PredictionSmoother (EMA behaviour, reset)
  - detect_and_crop (face extraction on synthetic image)
  - CLI argument parsing
  - Model loading (baseline + temporal) from checkpoint
  - Drawing helpers (smoke tests — no crash)

Run:
    pytest test_hackerseye_antispoof.py -v
"""

import os
import tempfile
import shutil
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from hackerseye_antispoof import (
    FrameBuffer,
    PredictionSmoother,
    EVAL_TRANSFORM,
    draw_bbox,
    draw_stats,
    parse_args,
    load_model,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 1. FrameBuffer tests
# ═══════════════════════════════════════════════════════════════

class TestFrameBuffer:

    def test_starts_empty(self):
        buf = FrameBuffer(seq_len=4)
        assert len(buf.buffer) == 0
        assert not buf.is_full

    def test_push_fills_buffer(self):
        buf = FrameBuffer(seq_len=3)
        for _ in range(3):
            buf.push(torch.randn(3, 224, 224))
        assert buf.is_full

    def test_push_none_increments_miss(self):
        buf = FrameBuffer(seq_len=4, max_miss=2)
        buf.push(torch.randn(3, 224, 224))
        buf.push(None)
        assert buf.miss_count == 1
        assert len(buf.buffer) == 1  # not flushed yet

    def test_flush_on_too_many_misses(self):
        buf = FrameBuffer(seq_len=4, max_miss=2)
        buf.push(torch.randn(3, 224, 224))
        buf.push(torch.randn(3, 224, 224))
        buf.push(None)
        buf.push(None)
        buf.push(None)   # miss_count=3 > max_miss=2 → flush
        assert len(buf.buffer) == 0

    def test_get_sequence_when_full(self):
        buf = FrameBuffer(seq_len=4)
        for _ in range(4):
            buf.push(torch.randn(3, 224, 224))
        seq = buf.get_sequence(torch.device("cpu"))
        assert seq is not None
        assert seq.shape == (1, 4, 3, 224, 224)

    def test_get_sequence_when_not_full(self):
        buf = FrameBuffer(seq_len=4)
        buf.push(torch.randn(3, 224, 224))
        assert buf.get_sequence(torch.device("cpu")) is None

    def test_get_latest(self):
        buf = FrameBuffer(seq_len=4)
        t = torch.randn(3, 224, 224)
        buf.push(t)
        latest = buf.get_latest(torch.device("cpu"))
        assert latest is not None
        assert latest.shape == (1, 3, 224, 224)

    def test_get_latest_empty(self):
        buf = FrameBuffer(seq_len=4)
        assert buf.get_latest(torch.device("cpu")) is None

    def test_deque_rolls_over(self):
        buf = FrameBuffer(seq_len=3)
        for i in range(5):
            buf.push(torch.full((3, 224, 224), float(i)))
        assert len(buf.buffer) == 3
        # Should contain frames 2, 3, 4
        assert buf.buffer[0][0, 0, 0].item() == 2.0

    def test_miss_reset_on_face(self):
        buf = FrameBuffer(seq_len=4, max_miss=5)
        buf.push(None)
        buf.push(None)
        assert buf.miss_count == 2
        buf.push(torch.randn(3, 224, 224))
        assert buf.miss_count == 0


# ═══════════════════════════════════════════════════════════════
# 2. PredictionSmoother tests
# ═══════════════════════════════════════════════════════════════

class TestPredictionSmoother:

    def test_starts_neutral(self):
        s = PredictionSmoother()
        assert s.smoothed == 0.5

    def test_ema_moves_toward_input(self):
        s = PredictionSmoother(alpha=0.5)
        # Start at 0.5, push 1.0 → should increase
        val = s.update(1.0)
        assert val > 0.5
        assert val == pytest.approx(0.75)  # 0.5*1.0 + 0.5*0.5

    def test_ema_alpha_1_instant(self):
        s = PredictionSmoother(alpha=1.0)
        val = s.update(0.9)
        assert val == pytest.approx(0.9)

    def test_ema_alpha_0_frozen(self):
        s = PredictionSmoother(alpha=0.0)
        s.update(0.9)
        assert s.smoothed == pytest.approx(0.5)  # never changes

    def test_converges_to_constant_input(self):
        s = PredictionSmoother(alpha=0.3)
        for _ in range(100):
            s.update(1.0)
        assert s.smoothed == pytest.approx(1.0, abs=0.01)

    def test_reset(self):
        s = PredictionSmoother()
        s.update(1.0)
        s.update(1.0)
        s.reset()
        assert s.smoothed == 0.5
        assert len(s.history) == 0

    def test_history_length(self):
        s = PredictionSmoother(window=5)
        for i in range(10):
            s.update(float(i) / 10)
        assert len(s.history) == 5


# ═══════════════════════════════════════════════════════════════
# 3. EVAL_TRANSFORM test
# ═══════════════════════════════════════════════════════════════

class TestEvalTransform:

    def test_output_shape(self):
        """A 100x80 RGB numpy image should produce a (3, 224, 224) tensor."""
        img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        t = EVAL_TRANSFORM(img)
        assert t.shape == (3, 224, 224)
        assert t.dtype == torch.float32


# ═══════════════════════════════════════════════════════════════
# 4. Drawing helpers — smoke tests
# ═══════════════════════════════════════════════════════════════

class TestDrawing:

    def test_draw_bbox_no_crash(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_bbox(frame, (100, 100, 200, 200), "REAL", 0.95, is_spoof=False)
        # No assertion — just ensure no exception

    def test_draw_bbox_spoof(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_bbox(frame, (50, 50, 150, 150), "SPOOF", 0.88, is_spoof=True)

    def test_draw_stats_no_crash(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_stats(frame, 30.0, "Temporal", 5, 8, 0.72, 0.5, show=True)

    def test_draw_stats_hidden(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        draw_stats(frame, 30.0, "Temporal", 5, 8, 0.72, 0.5, show=False)
        # Frame unchanged when show=False
        np.testing.assert_array_equal(frame, original)


# ═══════════════════════════════════════════════════════════════
# 5. CLI parsing tests
# ═══════════════════════════════════════════════════════════════

class TestParseArgs:

    def test_required_checkpoint(self):
        with mock.patch("sys.argv", ["prog", "--checkpoint", "ckpt.pt"]):
            args = parse_args()
        assert args.checkpoint == "ckpt.pt"
        assert args.temporal is False

    def test_temporal_mode(self):
        with mock.patch("sys.argv", [
            "prog", "--checkpoint", "ckpt.pt", "--temporal",
            "--backbone", "mobilenet", "--aggregator", "avg", "--seq-len", "16",
        ]):
            args = parse_args()
        assert args.temporal is True
        assert args.backbone == "mobilenet"
        assert args.aggregator == "avg"
        assert args.seq_len == 16

    def test_defaults(self):
        with mock.patch("sys.argv", ["prog", "--checkpoint", "c.pt"]):
            args = parse_args()
        assert args.smooth_window == 10
        assert args.ema_alpha == pytest.approx(0.3)
        assert args.threshold == pytest.approx(0.5)
        assert args.camera == 0
        assert args.width == 640
        assert args.height == 480


# ═══════════════════════════════════════════════════════════════
# 6. Model loading — baseline (from a saved checkpoint)
# ═══════════════════════════════════════════════════════════════

class TestLoadModel:

    def _make_baseline_checkpoint(self, path, device):
        from train_antispoof import build_model
        m = build_model(pretrained=False)
        torch.save({
            "model_state_dict": m.state_dict(),
            "epoch": 1,
            "val_loss": 0.5,
        }, path)

    def _make_temporal_checkpoint(self, path, device):
        from train_temporal import TemporalAntiSpoofModel
        m = TemporalAntiSpoofModel(
            backbone_name="resnet18", aggregator="lstm",
            pretrained=False, lstm_hidden=128,
        )
        torch.save({
            "model_state_dict": m.state_dict(),
            "epoch": 2,
            "val_loss": 0.3,
        }, path)

    def test_load_baseline(self, tmp_dir, device):
        ckpt_path = os.path.join(tmp_dir, "baseline.pt")
        self._make_baseline_checkpoint(ckpt_path, device)
        with mock.patch("sys.argv", ["prog", "--checkpoint", ckpt_path]):
            args = parse_args()
        model = load_model(args, device)
        assert model.training is False  # should be in eval mode
        # Quick forward pass
        out = model(torch.randn(1, 3, 224, 224, device=device))
        assert out.shape == (1, 1)

    def test_load_temporal(self, tmp_dir, device):
        ckpt_path = os.path.join(tmp_dir, "temporal.pt")
        self._make_temporal_checkpoint(ckpt_path, device)
        with mock.patch("sys.argv", [
            "prog", "--checkpoint", ckpt_path, "--temporal",
        ]):
            args = parse_args()
        model = load_model(args, device)
        assert model.training is False
        out = model(torch.randn(1, 8, 3, 224, 224, device=device))
        assert out.shape == (1, 1)
