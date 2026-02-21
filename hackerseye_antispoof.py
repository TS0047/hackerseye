"""
hackerseye_antispoof.py — Phase 8: Live webcam anti-spoofing inference.

Real-time face detection + anti-spoofing classification using a trained
Phase 3 (baseline CNN) or Phase 4 (temporal) checkpoint.

Pipeline per frame:
  1. Capture webcam frame (OpenCV).
  2. Detect face with MTCNN and crop to 224×224.
  3. Buffer the last N crops for temporal smoothing.
  4. Run inference through the loaded model (single-frame or temporal).
  5. Smooth predictions over the buffer (exponential moving average).
  6. Display bounding box, label (REAL / SPOOF), and confidence live.

Controls:
    Q  — quit
    S  — toggle stats overlay
    M  — cycle model mode (single / temporal)

Usage:
    # With a Phase 3 baseline checkpoint (single-frame):
    python hackerseye_antispoof.py --checkpoint checkpoints/best_antispoof.pt

    # With a Phase 4 temporal checkpoint (sequence of 8 frames):
    python hackerseye_antispoof.py --checkpoint checkpoints/best_temporal.pt \\
                                   --temporal --seq-len 8

    # Choose backbone (must match what the checkpoint was trained with):
    python hackerseye_antispoof.py --checkpoint checkpoints/best_temporal.pt \\
                                   --temporal --backbone mobilenet
"""

# ═══════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════
import argparse
import os
import sys
import time
import logging
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

# ── MTCNN for face detection ────────────────────────────────
from facenet_pytorch import MTCNN

# ── Our models ──────────────────────────────────────────────
from train_antispoof import build_model as build_baseline_model
from train_temporal import TemporalAntiSpoofModel
from antispoof_dataset import IMAGENET_MEAN, IMAGENET_STD

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
# 1. CLI arguments
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 8 — Live anti-spoofing demo with temporal smoothing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained model checkpoint (.pt).")
    p.add_argument("--temporal", action="store_true",
                   help="Use Phase 4 temporal model instead of Phase 3 baseline.")
    p.add_argument("--backbone", choices=["resnet18", "mobilenet"], default="resnet18",
                   help="CNN backbone (must match checkpoint).")
    p.add_argument("--aggregator", choices=["avg", "max", "lstm"], default="lstm",
                   help="Temporal aggregator (only for --temporal).")
    p.add_argument("--seq-len", type=int, default=8,
                   help="Sequence length for temporal model.")
    p.add_argument("--smooth-window", type=int, default=10,
                   help="Number of recent predictions to smooth over (EMA buffer).")
    p.add_argument("--ema-alpha", type=float, default=0.3,
                   help="EMA smoothing factor (0–1). Higher = more responsive.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold on smoothed probability (>= threshold → spoof).")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera device index.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# 2. Model loading
# ═══════════════════════════════════════════════════════════════

def load_model(args, device: torch.device) -> nn.Module:
    """
    Instantiate and load the correct model from a checkpoint.

    For Phase 3 (--temporal not set):
      - build_baseline_model() → ResNet-18 with Linear(512,1) head
    For Phase 4 (--temporal):
      - TemporalAntiSpoofModel with chosen backbone + aggregator

    The checkpoint stores 'model_state_dict' which we load in here.
    """
    if args.temporal:
        logger.info("Building temporal model: backbone=%s, aggregator=%s, seq_len=%d",
                     args.backbone, args.aggregator, args.seq_len)
        model = TemporalAntiSpoofModel(
            backbone_name=args.backbone,
            aggregator=args.aggregator,
            pretrained=False,           # we're loading trained weights
            freeze_backbone=False,
        )
    else:
        logger.info("Building baseline (single-frame) model.")
        model = build_baseline_model(pretrained=False)

    # ── Load checkpoint weights ──────────────────────────────
    if not os.path.isfile(args.checkpoint):
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    logger.info("Loaded checkpoint: epoch=%s  val_loss=%.4f", epoch, val_loss)
    return model


# ═══════════════════════════════════════════════════════════════
# 3. Frame processing — MTCNN face detection + crop
# ═══════════════════════════════════════════════════════════════
#
# FRAME PROCESSING PIPELINE:
#   1. OpenCV captures a BGR frame from the webcam.
#   2. Convert BGR → RGB (MTCNN expects RGB).
#   3. MTCNN detects face bounding boxes + confidence.
#   4. Take the highest-confidence face.
#   5. Crop the face region from the RGB frame.
#   6. Resize to 224×224 (matching training-time input size).
#   7. Apply evaluation transforms (ToTensor + ImageNet normalise).
#   8. Return the tensor + bounding box for drawing.
#

# Evaluation transform: resize → tensor → ImageNet normalise
# (deterministic — no augmentation at inference time)
EVAL_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def detect_and_crop(
    frame_bgr: np.ndarray,
    mtcnn: MTCNN,
) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int, int, int]]]:
    """
    Detect the primary face in a BGR frame and return a cropped 224×224 tensor.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Raw BGR frame from OpenCV.
    mtcnn : MTCNN
        Initialised MTCNN detector.

    Returns
    -------
    Tuple[Optional[Tensor], Optional[Tuple]]
        (face_tensor [3,224,224], bbox (x1,y1,x2,y2))  or (None, None) if no face.
    """
    # Convert BGR → RGB for MTCNN
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Detect faces — returns boxes (N,4) and probs (N,)
    boxes, probs = mtcnn.detect(frame_rgb)

    if boxes is None or len(boxes) == 0:
        return None, None

    # Pick the highest-confidence detection
    best_idx = int(np.argmax(probs))
    x1, y1, x2, y2 = boxes[best_idx]

    # Clamp to frame bounds
    h, w = frame_rgb.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    # Sanity check — face must be a reasonable size
    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None, None

    # Crop and transform
    face_rgb = frame_rgb[y1:y2, x1:x2]
    face_tensor = EVAL_TRANSFORM(face_rgb)  # (3, 224, 224)

    return face_tensor, (x1, y1, x2, y2)


# ═══════════════════════════════════════════════════════════════
# 4. Sequence buffer — maintains a rolling window of face crops
# ═══════════════════════════════════════════════════════════════
#
# SEQUENCE BUFFERING:
#   For the temporal model (Phase 4), we need T consecutive face crops.
#   We maintain a deque of max length = seq_len.  Each time a new face
#   is detected, it's appended to the buffer.
#
#   When the buffer is full (len == seq_len), we stack all T tensors
#   into a (1, T, 3, 224, 224) batch and run temporal inference.
#
#   If the buffer isn't full yet, we fall back to single-frame inference
#   on the latest crop (or show "Buffering…" on screen).
#
#   The buffer is cleared if no face is detected for several consecutive
#   frames (to avoid mixing crops from different people / scenes).
#

class FrameBuffer:
    """
    Rolling buffer of face crop tensors for temporal inference.

    Attributes
    ----------
    buffer : deque
        Fixed-size deque holding the most recent face tensors.
    miss_count : int
        Consecutive frames with no face detected.  If this exceeds
        `max_miss`, the buffer is flushed.
    """

    def __init__(self, seq_len: int, max_miss: int = 5):
        self.seq_len = seq_len
        self.max_miss = max_miss
        self.buffer: deque = deque(maxlen=seq_len)
        self.miss_count = 0

    def push(self, face_tensor: Optional[torch.Tensor]) -> None:
        """Add a new face crop, or register a miss."""
        if face_tensor is not None:
            self.buffer.append(face_tensor)
            self.miss_count = 0
        else:
            self.miss_count += 1
            if self.miss_count > self.max_miss:
                self.buffer.clear()

    @property
    def is_full(self) -> bool:
        return len(self.buffer) == self.seq_len

    def get_sequence(self, device: torch.device) -> Optional[torch.Tensor]:
        """
        Stack buffer into a (1, T, C, H, W) tensor for temporal inference.

        Returns None if the buffer isn't full yet.
        """
        if not self.is_full:
            return None
        # Stack: list of (3,224,224) → (T,3,224,224) → (1,T,3,224,224)
        seq = torch.stack(list(self.buffer), dim=0).unsqueeze(0)
        return seq.to(device)

    def get_latest(self, device: torch.device) -> Optional[torch.Tensor]:
        """Get the most recent crop as (1,3,224,224) for single-frame fallback."""
        if len(self.buffer) == 0:
            return None
        return self.buffer[-1].unsqueeze(0).to(device)


# ═══════════════════════════════════════════════════════════════
# 5. Prediction smoother — exponential moving average
# ═══════════════════════════════════════════════════════════════
#
# INFERENCE LOGIC — TEMPORAL SMOOTHING:
#   Raw per-frame predictions can be noisy: a single frame might flicker
#   between real/spoof.  We smooth the output probability using an
#   exponential moving average (EMA):
#
#       smoothed[t] = α · raw[t]  +  (1 − α) · smoothed[t−1]
#
#   where α (ema_alpha) controls responsiveness.  α ≈ 0.3 gives a nice
#   balance between stability and latency.
#
#   The smoothed probability is then compared to a decision threshold
#   (default 0.5) to produce the final REAL / SPOOF label.
#

class PredictionSmoother:
    """
    Exponential moving average smoother for anti-spoof predictions.

    Attributes
    ----------
    alpha : float
        EMA weight for the newest observation.  Higher = more reactive.
    smoothed : float
        Current smoothed spoof probability (0 = certainly real, 1 = certainly spoof).
    history : deque
        Ring buffer of raw probabilities for optional stats.
    """

    def __init__(self, alpha: float = 0.3, window: int = 10):
        self.alpha = alpha
        self.smoothed = 0.5         # start neutral
        self.history: deque = deque(maxlen=window)

    def update(self, raw_prob: float) -> float:
        """
        Push a new raw spoof probability and return the smoothed value.

        Parameters
        ----------
        raw_prob : float
            Sigmoid output from the model (0–1).  Higher = more spoof-like.

        Returns
        -------
        float
            Smoothed spoof probability.
        """
        self.history.append(raw_prob)
        self.smoothed = self.alpha * raw_prob + (1.0 - self.alpha) * self.smoothed
        return self.smoothed

    def reset(self) -> None:
        self.smoothed = 0.5
        self.history.clear()


# ═══════════════════════════════════════════════════════════════
# 6. Drawing helpers
# ═══════════════════════════════════════════════════════════════

def draw_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    is_spoof: bool,
) -> None:
    """Draw a bounding box with label + confidence on the frame."""
    x1, y1, x2, y2 = bbox
    color = (0, 0, 255) if is_spoof else (0, 220, 0)    # Red vs Green

    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background + text
    text = f"{label}  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    label_y = max(y1 - 10, th + 10)
    cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 8, label_y + 4), color, -1)
    cv2.putText(frame, text, (x1 + 4, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def draw_stats(
    frame: np.ndarray,
    fps: float,
    mode: str,
    buf_fill: int,
    buf_max: int,
    smoothed: float,
    threshold: float,
    show: bool,
) -> None:
    """Draw an overlay with FPS, mode, buffer fill, and smoothed score."""
    if not show:
        return

    lines = [
        f"FPS:       {fps:.1f}",
        f"Mode:      {mode}",
        f"Buffer:    {buf_fill}/{buf_max}",
        f"Smoothed:  {smoothed:.3f}",
        f"Threshold: {threshold}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
# 7. Main inference loop
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Device ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load model ───────────────────────────────────────────
    model = load_model(args, device)
    mode_label = "Temporal" if args.temporal else "Single-frame"

    # ── MTCNN face detector ──────────────────────────────────
    # keep_all=False: only return the highest-confidence face
    # device: run MTCNN on same device as the model
    mtcnn = MTCNN(
        keep_all=False,
        device=device,
        select_largest=True,
        post_process=False,
    )
    logger.info("MTCNN initialised on %s.", device)

    # ── Frame buffer for temporal inference ───────────────────
    frame_buf = FrameBuffer(seq_len=args.seq_len)

    # ── Prediction smoother ──────────────────────────────────
    smoother = PredictionSmoother(alpha=args.ema_alpha, window=args.smooth_window)

    # ── Webcam ───────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        logger.error("Cannot open camera %d.", args.camera)
        sys.exit(1)

    # ── FPS tracking ─────────────────────────────────────────
    fps_counter = 0
    fps_display = 0.0
    fps_timer = time.time()
    show_stats_flag = True

    logger.info("=" * 55)
    logger.info("HackersEye Anti-Spoof Live Demo")
    logger.info("  Mode:      %s", mode_label)
    logger.info("  Smoothing: EMA α=%.2f  window=%d", args.ema_alpha, args.smooth_window)
    logger.info("  Threshold: %.2f", args.threshold)
    logger.info("  Q=quit  S=stats")
    logger.info("=" * 55)

    # ══════════════════════════════════════════════════════════
    # Main loop
    # ══════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Frame capture failed.")
            break

        # ── 1. Detect + crop face ────────────────────────────
        face_tensor, bbox = detect_and_crop(frame, mtcnn)

        # ── 2. Push into sequence buffer ─────────────────────
        frame_buf.push(face_tensor)

        # ── 3. Inference ─────────────────────────────────────
        #
        # INFERENCE LOGIC:
        #   - If temporal mode AND buffer is full:
        #       Stack T frames → (1,T,C,H,W) → temporal model → logit
        #   - Else if we have at least one face crop:
        #       Use latest crop → (1,C,H,W) → baseline model → logit
        #       (works even in temporal mode as a warm-up fallback)
        #   - Else: no face → skip inference, keep previous smoothed value
        #
        raw_prob = None

        with torch.no_grad():
            if args.temporal and frame_buf.is_full:
                # ── Temporal inference ───────────────────────
                seq = frame_buf.get_sequence(device)   # (1, T, C, H, W)
                logit = model(seq).squeeze()            # scalar
                raw_prob = torch.sigmoid(logit).item()

            elif face_tensor is not None and not args.temporal:
                # ── Single-frame inference ───────────────────
                inp = face_tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)
                logit = model(inp).squeeze()
                raw_prob = torch.sigmoid(logit).item()

            elif face_tensor is not None and args.temporal and not frame_buf.is_full:
                # ── Temporal warmup: not enough frames yet ───
                # Show "Buffering" — don't predict until we have seq_len frames
                pass

        # ── 4. Smooth the prediction ─────────────────────────
        if raw_prob is not None:
            smoothed = smoother.update(raw_prob)
        else:
            smoothed = smoother.smoothed   # hold previous value

        # ── 5. Decision: REAL or SPOOF ───────────────────────
        is_spoof = smoothed >= args.threshold
        label_text = "SPOOF" if is_spoof else "REAL"
        confidence = smoothed if is_spoof else (1.0 - smoothed)

        # ── 6. Draw results ──────────────────────────────────
        if bbox is not None:
            draw_bbox(frame, bbox, label_text, confidence, is_spoof)
        elif face_tensor is None:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

        # Buffering indicator for temporal warm-up
        if args.temporal and not frame_buf.is_full and face_tensor is not None:
            fill = len(frame_buf.buffer)
            cv2.putText(frame, f"Buffering... {fill}/{frame_buf.seq_len}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # ── FPS counter ──────────────────────────────────────
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()

        draw_stats(frame, fps_display, mode_label,
                   len(frame_buf.buffer), frame_buf.seq_len,
                   smoothed, args.threshold, show_stats_flag)

        cv2.putText(frame, "Q=Quit  S=Stats",
                    (10, args.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("HackersEye Anti-Spoof", frame)

        # ── Key handling ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            logger.info("Quit by user.")
            break
        elif key == ord('s'):
            show_stats_flag = not show_stats_flag

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
